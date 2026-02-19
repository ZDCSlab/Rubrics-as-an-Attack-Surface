#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


# -------------------------
# Utilities: content parsing
# -------------------------

def _content_to_text(content: Any) -> str:
    """
    HF message content formats:
    - string
    - list of dicts like [{"type":"text","text":"..."}]
    - dict
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join([p for p in parts if p.strip()])
    if isinstance(content, dict):
        txt = content.get("text")
        return txt if isinstance(txt, str) else ""
    return str(content)


def _msg_to_text(msg: Dict[str, Any]) -> str:
    body = _content_to_text(msg.get("content"))

    # webdev sometimes stores code here
    obj = msg.get("object")
    if isinstance(obj, dict):
        code = obj.get("code")
        if isinstance(code, str) and code.strip():
            if body.strip():
                body = body.rstrip() + "\n\n" + code
            else:
                body = code

    return body.strip()


def _is_single_turn(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) != 2:
        return False
    r0 = messages[0].get("role")
    r1 = messages[1].get("role")
    return (r0 == "user") and (r1 == "assistant")


def _get_uid(row: Dict[str, Any], dataset: str, instruction: str = None) -> str:
    # Prefer native id-like fields
    for k in ["id", "question_id", "pair_id", "sample_id"]:
        v = row.get(k)
        if v is not None:
            return f"{dataset}:{k}:{v}"

    # Fallback: stable-ish hash of key fields we rely on
    payload = {"instruction": instruction}
    s = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return f"{dataset}:md5:{h}"


def _count_tokens(tokenizer, text: str) -> int:
    # No special tokens: count approximate context usage
    return len(tokenizer.encode(text, add_special_tokens=False))


def _joined_text_for_len(instruction: str, resp1: str, resp2: str) -> str:
    # Approximates real judge prompt better than raw concatenation.
    return (
        instruction.strip()
        + "\n\n[Response A]\n" + resp1.strip()
        + "\n\n[Response B]\n" + resp2.strip()
    )


# -------------------------
# Extractor: UltraFeedback Binarized
# -------------------------

def extract_ultrafeedback_binarized(
    row: Dict[str, Any],
    dataset: str,
    tokenizer,
    max_total_tokens: int
) -> Tuple[Optional[Dict[str, Any]], bool, bool, Optional[str]]:
    """
    For HuggingFaceH4/ultrafeedback_binarized (prefs splits):
      - prompt: str
      - prompt_id: str
      - chosen:  [{"role":"user","content":...}, {"role":"assistant","content":...}]
      - rejected:[{"role":"user","content":...}, {"role":"assistant","content":...}]
    We create pairwise A/B with deterministic 50% swap.
    Returns:
      item, base_ok, len_ok, gt
    """
    chosen = row.get("chosen")
    rejected = row.get("rejected")

    if not (_is_single_turn(chosen) and _is_single_turn(rejected)):
        return None, False, False, None

    instruction = (row.get("prompt") or _msg_to_text(chosen[0]) or "").strip()
    resp_chosen = _msg_to_text(chosen[1]).strip()
    resp_rejected = _msg_to_text(rejected[1]).strip()

    if not instruction or not resp_chosen or not resp_rejected:
        return None, False, False, None

    # Stable UID (use prompt_id if present)
    pid = row.get("prompt_id")
    uid = f"{dataset}:prompt_id:{pid}" if pid else _get_uid(row, dataset, instruction)

    # Default: chosen is better => A wins
    resp1, resp2 = resp_chosen, resp_rejected
    gt = "A"

    # Deterministic 50% swap to enable A/B balancing without RNG bias
    h = hashlib.md5(uid.encode("utf-8")).hexdigest()
    if (int(h[-1], 16) % 2) == 1:
        resp1, resp2 = resp2, resp1
        gt = "B"

    base_ok = True
    joined = _joined_text_for_len(instruction, resp1, resp2)
    tok = _count_tokens(tokenizer, joined)
    len_ok = tok <= max_total_tokens

    if not len_ok:
        return None, base_ok, False, gt

    item = {
        "uid": uid,
        "dataset": dataset,
        "instruction": instruction,
        "response1": resp1,
        "response2": resp2,
        "ground_truth": gt,  # 'A' or 'B'
        "token_len": tok,
    }
    return item, base_ok, True, gt


# -------------------------
# One-pass balanced reservoir -> multi-splits (NO OVERLAP)
# -------------------------

def balanced_reservoir_stream_for_files(
    iterable,
    extractor_fn,
    dataset_name: str,
    tokenizer,
    max_total_tokens: int,
    split_sizes: Dict[str, int],   # {"train":1000,"val":1000,"test":1000,"dpo_20k":20000,"dpo_1k":1000}
    seed: int,
    show_progress: bool = True,
):
    """
    One-pass streaming, class-conditional (A/B) balanced reservoir sampling,
    then cut into multiple splits according to split_sizes.

    Guarantees:
      - each split is exactly 50/50 A/B (requires even split sizes)
      - no uid overlap across splits
      - uid de-dup inside accepted pool
    """
    rng = random.Random(seed)

    # validate split sizes
    for k, n in split_sizes.items():
        if n % 2 != 0:
            raise ValueError(f"split '{k}' must be even for A/B balance, got {n}")

    total_needed = sum(split_sizes.values())
    needA = total_needed // 2
    needB = total_needed - needA

    resA: List[Dict[str, Any]] = []
    resB: List[Dict[str, Any]] = []

    base_total = 0
    len_total = 0
    base_by = {"A": 0, "B": 0}
    len_by = {"A": 0, "B": 0}
    seen_len_by = {"A": 0, "B": 0}

    seen_uids = set()

    it = iterable
    if show_progress:
        it = tqdm(it, desc=f"{dataset_name} [sampling]", unit="rows", miniters=200)

    for row in it:
        item, base_ok, len_ok, gt = extractor_fn(row, dataset_name, tokenizer, max_total_tokens)

        if base_ok:
            base_total += 1
            if gt in ("A", "B"):
                base_by[gt] += 1

        if not len_ok or gt not in ("A", "B") or item is None:
            if show_progress:
                it.set_postfix({
                    "base": base_total,
                    "len_ok": len_total,
                    "lenA": len_by["A"],
                    "lenB": len_by["B"],
                    "keepA": len(resA),
                    "keepB": len(resB),
                }, refresh=True)
            continue

        uid = item["uid"]
        if uid in seen_uids:
            if show_progress:
                it.set_postfix({
                    "base": base_total,
                    "len_ok": len_total,
                    "lenA": len_by["A"],
                    "lenB": len_by["B"],
                    "keepA": len(resA),
                    "keepB": len(resB),
                    "dup": len(seen_uids),
                })
            continue
        seen_uids.add(uid)

        len_total += 1
        len_by[gt] += 1
        seen_len_by[gt] += 1

        # class-conditional reservoir
        if gt == "A":
            if len(resA) < needA:
                resA.append(item)
            else:
                j = rng.randrange(seen_len_by["A"])
                if j < needA:
                    resA[j] = item
        else:
            if len(resB) < needB:
                resB.append(item)
            else:
                j = rng.randrange(seen_len_by["B"])
                if j < needB:
                    resB[j] = item

        if show_progress:
            it.set_postfix({
                "base": base_total,
                "len_ok": len_total,
                "lenA": len_by["A"],
                "lenB": len_by["B"],
                "keepA": len(resA),
                "keepB": len(resB),
            })

        # optional early stop: once reservoirs full, we COULD stop, but we keep going for better randomness.
        # If you want speed, uncomment:
        # if len(resA) >= needA and len(resB) >= needB:
        #     break

    if len(resA) < needA or len(resB) < needB:
        raise RuntimeError(
            f"filtered [{dataset_name}] does not have enough A/B balanced data.\n"
            f"need A={needA}, B={needB}; have A={len(resA)}, B={len(resB)}.\n"
            f"len_ok data number: A={len_by['A']}, B={len_by['B']}."
        )

    rng.shuffle(resA)
    rng.shuffle(resB)

    # build splits by slicing without overlap
    splits: Dict[str, List[Dict[str, Any]]] = {}
    offA = 0
    offB = 0

    # preserve insertion order of dict in py3.7+
    for split_name, n in split_sizes.items():
        nA = n // 2
        nB = n // 2
        part = resA[offA:offA + nA] + resB[offB:offB + nB]
        offA += nA
        offB += nB
        rng.shuffle(part)
        splits[split_name] = part

    # sanity: no overlap across splits
    all_uids: List[str] = []
    for _, rows in splits.items():
        all_uids.extend([x["uid"] for x in rows])
    if len(set(all_uids)) != len(all_uids):
        raise RuntimeError(f"[{dataset_name}] repeated uid across splits (should not happen).")

    stats = {
        "dataset": dataset_name,
        "base_total": base_total,
        "base_A": base_by["A"],
        "base_B": base_by["B"],
        "len_total": len_total,
        "len_A": len_by["A"],
        "len_B": len_by["B"],
        "need_A": needA,
        "need_B": needB,
        "splits": {k: len(v) for k, v in splits.items()},
        "splits_A": {k: sum(1 for x in v if x["ground_truth"] == "A") for k, v in splits.items()},
        "splits_B": {k: sum(1 for x in v if x["ground_truth"] == "B") for k, v in splits.items()},
    }
    return splits, stats


def save_jsonl(path: Path, rows: List[Dict[str, Any]], split_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            out = {
                "uid": r["uid"],
                "dataset": r["dataset"],
                "split": split_name,
                "instruction": r["instruction"],
                "response_a": r["response1"],
                "response_b": r["response2"],
                "ground_truth": r["ground_truth"],  # 'A' or 'B'
                "token_len": r["token_len"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-8B",
                    help="Tokenizer name/path used to count tokens (ideally same family as your judge).")
    ap.add_argument("--max-total-tokens", type=int, default=4096,
                    help="Max tokens allowed for concatenated instruction+respA+respB (default=4096).")

    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="./data/helpfulness",
                    help="Output folder prefix.")
    ap.add_argument("--dpo", action="store_true",
                    help="Also export dpo_20k.jsonl (20000) and dpo_1k.jsonl (1000), non-overlap with train/val/test.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = ap.parse_args()

    # require even for strict A/B balance
    for name, n in [("n_train", args.n_train), ("n_val", args.n_val), ("n_test", args.n_test)]:
        if n % 2 != 0:
            raise ValueError(f"--{name.replace('_','-')} must be even for A/B balance, got {n}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_name = "HuggingFaceH4/ultrafeedback_binarized"
    split = "train_prefs"
    short = "ultrafeedback"

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Max total tokens: {args.max_total_tokens}")
    print(f"Seed: {args.seed}")
    print(f"Dataset: {ds_name} [{split}]")
    print(f"Split sizes: train={args.n_train}, val={args.n_val}, test={args.n_test}")
    if args.dpo:
        print("DPO: enabled (dpo_20k=20000, dpo_1k=1000) [non-overlap guaranteed]")
    print("")

    # Build split size plan in ONE PASS to guarantee no overlap
    split_sizes = {
        "train": args.n_train,
        "val": args.n_val,
        "test": args.n_test,
    }
    if args.dpo:
        split_sizes["dpo_20k"] = 20000
        split_sizes["dpo_1k"] = 1000

    print(f"=== Loading {ds_name} [{split}] (streaming) ===")
    ds = load_dataset(ds_name, split=split, streaming=True)

    splits, stats = balanced_reservoir_stream_for_files(
        iterable=ds,
        extractor_fn=extract_ultrafeedback_binarized,
        dataset_name=ds_name,
        tokenizer=tokenizer,
        max_total_tokens=args.max_total_tokens,
        split_sizes=split_sizes,
        seed=args.seed,
        show_progress=(not args.no_progress),
    )

    # Print stats
    print(f"[{ds_name}] base_ok total: {stats['base_total']} (A={stats['base_A']}, B={stats['base_B']})")
    print(f"[{ds_name}] len_ok  total: {stats['len_total']}  (A={stats['len_A']}, B={stats['len_B']})")
    for k in split_sizes.keys():
        print(f"[{ds_name}] SPLIT {k:<7}={stats['splits'][k]} (A/B={stats['splits_A'][k]}/{stats['splits_B'][k]})")

    # Save files
    output_dir = Path(args.out_prefix, short)
    output_dir.mkdir(parents=True, exist_ok=True)

    name_to_file = {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
        "dpo_20k": "dpo_20k.jsonl",
        "dpo_1k": "dpo_1k.jsonl",
    }

    print("\n=== Writing ===")
    for split_name, rows in splits.items():
        filename = name_to_file.get(split_name, f"{split_name}.jsonl")
        out_path = output_dir / filename
        save_jsonl(out_path, rows, split_name)
        print(f"wrote: {out_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
