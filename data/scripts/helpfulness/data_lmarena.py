#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple
import argparse
import hashlib
import json
import random

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm


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


def _winner_to_gt(winner: Any) -> Optional[str]:
    if winner == "model_a":
        return "A"
    if winner == "model_b":
        return "B"
    return None  # tie / both_bad / others -> drop


def _get_uid(row: Dict[str, Any], dataset: str, instruction: str = None) -> str:
    # Prefer native id-like fields
    for k in ["id", "question_id", "pair_id", "sample_id"]:
        v = row.get(k)
        if v is not None:
            return f"{dataset}:{k}:{v}"

    # Fallback: strong-ish hash using instruction only
    payload = {"instruction": instruction}
    s = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return f"{dataset}:md5:{h}"


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _joined_text_for_len(instruction: str, resp1: str, resp2: str) -> str:
    return (
        instruction.strip()
        + "\n\n[Response A]\n" + resp1.strip()
        + "\n\n[Response B]\n" + resp2.strip()
    )


def _get_category_bool(row: Dict[str, Any], group_key: str, field: str) -> Optional[bool]:
    """
    row["category_tag"][group_key][field] -> bool
    Return True/False if found, else None.
    """
    ct = row.get("category_tag")
    if not isinstance(ct, dict):
        return None
    g = ct.get(group_key)
    if not isinstance(g, dict):
        return None
    v = g.get(field)
    return v if isinstance(v, bool) else None


# -------------------------
# Extractor factory
# -------------------------

def make_extract_arena140k_criteria(
    criteria_field: str,
    criteria_value: bool = True,
    require_creative_writing: Optional[bool] = None,
    require_math: Optional[bool] = None,
    require_is_code: Optional[bool] = None,
) -> Callable:
    """
    Returns extractor(row, dataset, tokenizer, max_total_tokens) -> (item, base_ok, len_ok, gt)
    """
    def _extract(row: Dict[str, Any], dataset: str, tokenizer, max_total_tokens: int):

        # 0) language
        v = row.get("language")
        if v is None or v != "en":
            return None, False, False, None

        # 1) criteria dimension
        if criteria_field != "skip":
            crit = _get_category_bool(row, "criteria_v0.1", criteria_field)
            if crit is None or crit != criteria_value:
                return None, False, False, None

        if require_creative_writing is not None:
            cw = _get_category_bool(row, "creative_writing_v0.1", "creative_writing")
            if cw is None or cw != require_creative_writing:
                return None, False, False, None

        if require_math is not None:
            m = _get_category_bool(row, "math_v0.1", "math")
            if m is None or m != require_math:
                return None, False, False, None

        if require_is_code is not None:
            ic = row.get("is_code")
            if ic is None or bool(ic) != bool(require_is_code):
                return None, False, False, None

        gt = _winner_to_gt(row.get("winner"))
        if gt is None:
            return None, False, False, None

        conv_a = row.get("conversation_a")
        conv_b = row.get("conversation_b")
        if not (_is_single_turn(conv_a) and _is_single_turn(conv_b)):
            return None, False, False, None

        instruction = _msg_to_text(conv_a[0])
        resp1 = _msg_to_text(conv_a[1])  # A
        resp2 = _msg_to_text(conv_b[1])  # B
        if not instruction or not resp1 or not resp2:
            return None, False, False, None

        joined = _joined_text_for_len(instruction, resp1, resp2)
        tok = _count_tokens(tokenizer, joined)
        len_ok = tok <= max_total_tokens
        if not len_ok:
            return None, True, False, gt

        item = {
            "uid": _get_uid(row, dataset, instruction),
            "dataset": dataset,
            "instruction": instruction,
            "response1": resp1,
            "response2": resp2,
            "ground_truth": gt,     # 'A' or 'B'
            "token_len": tok,
        }
        return item, True, True, gt

    return _extract


# -------------------------
# One-pass: balanced reservoir -> multi-splits (best-effort, no-overlap)
# -------------------------

def balanced_reservoir_stream_for_files_best_effort(
    iterable,
    extractor_fn,
    dataset_name: str,
    tokenizer,
    max_total_tokens: int,
    split_sizes: Dict[str, int],   # desired sizes per split
    seed: int,
    show_progress: bool = True,
):
    """
    Best-effort version:
      - tries to sample a balanced A/B pool up to total_needed, but if data is insufficient,
        will return as many as possible.
      - then allocates splits in priority order (insertion order of split_sizes),
        ensuring no overlap, and ensuring per-split 50/50 A/B *as much as possible*.

    Notes:
      - split sizes must be even to target perfect 50/50, but if insufficient, output may be smaller.
      - will NEVER throw due to insufficiency; only throws for invalid inputs.
    """
    rng = random.Random(seed)

    for k, n in split_sizes.items():
        if n % 2 != 0:
            raise ValueError(f"split '{k}' must be even (target A/B balance), got {n}")

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
                }, refresh=True)
            continue
        seen_uids.add(uid)

        len_total += 1
        len_by[gt] += 1
        seen_len_by[gt] += 1

        # class-conditional reservoir, capped at needA/needB (desired);
        # if insufficient, we'll just end with smaller pools.
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
            }, refresh=True)

    rng.shuffle(resA)
    rng.shuffle(resB)

    # Allocate splits in priority order (insertion order):
    # for each split: take min(available, target/2) from A and B.
    splits: Dict[str, List[Dict[str, Any]]] = {}
    offA = 0
    offB = 0

    for split_name, target_n in split_sizes.items():
        wantA = target_n // 2
        wantB = target_n // 2

        takeA = min(wantA, max(0, len(resA) - offA))
        takeB = min(wantB, max(0, len(resB) - offB))

        part = resA[offA:offA + takeA] + resB[offB:offB + takeB]
        offA += takeA
        offB += takeB

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
        "desired_need_A": needA,
        "desired_need_B": needB,
        "pool_A": len(resA),
        "pool_B": len(resB),
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
                "ground_truth": r["ground_truth"],
                "token_len": r["token_len"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-8B",
                    help="Tokenizer name/path used to count tokens (ideally same family as your judge).")
    ap.add_argument("--max-total-tokens", type=int, default=4096,
                    help="Max tokens allowed for concatenated instruction+resp1+resp2 (default=4096).")

    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="./data/helpfulness", help="Output file folder.")
    ap.add_argument("--dpo", action="store_true", help="Also save dpo_1k and dpo_20k when enabled (best-effort).")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = ap.parse_args()

    # enforce even for target A/B split; if not even, you can relax it but then A/B strict balance is impossible
    for name, n in [("n_train", args.n_train), ("n_val", args.n_val), ("n_test", args.n_test)]:
        if n % 2 != 0:
            raise ValueError(f"--{name.replace('_','-')} must be even for A/B balance target, got {n}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Max total tokens: {args.max_total_tokens}")
    print(f"Split sizes: train={args.n_train}, val={args.n_val}, test={args.n_test} (target A/B balanced)")
    print(f"Seed: {args.seed}")
    print(f"DPO flag: {args.dpo}")
    print("")

    # Extractors
    extract_arena_problem_solving = make_extract_arena140k_criteria(
        criteria_field="problem_solving",
        criteria_value=True,
    )
    extract_arena_real_world = make_extract_arena140k_criteria(
        criteria_field="real_world",
        criteria_value=True,
    )
    extract_arena_creative_writing = make_extract_arena140k_criteria(
        criteria_field="skip",
        criteria_value=True,
        require_creative_writing=True,
    )

    # jobs: (dataset_id, split, extractor_fn, short_dir, allow_dpo)
    jobs = [
        ("lmarena-ai/arena-human-preference-140k", "train", extract_arena_problem_solving, "problem_solving", False),  # default no dpo
        ("lmarena-ai/arena-human-preference-140k", "train", extract_arena_real_world, "real_world", True),
        ("lmarena-ai/arena-human-preference-140k", "train", extract_arena_creative_writing, "creative_writing", False), # default no dpo
    ]

    for ds_name, split, extractor, short, allow_dpo in jobs:
        print(f"=== Loading {ds_name} [{split}] (streaming) ===")

        # Build split plan with priority:
        # 1) train/val/test
        # 2) dpo_1k
        # 3) dpo_20k (lowest priority)
        split_sizes: Dict[str, int] = {
            "train": args.n_train,
            "val": args.n_val,
            "test": args.n_test,
        }

        do_dpo = bool(args.dpo and allow_dpo)
        if do_dpo:
            split_sizes["dpo_1k"] = 1000
            split_sizes["dpo_20k"] = 20000

        ds = load_dataset(ds_name, split=split, streaming=True)

        splits, stats = balanced_reservoir_stream_for_files_best_effort(
            iterable=ds,
            extractor_fn=extractor,
            dataset_name=f"{ds_name}:{short}",
            tokenizer=tokenizer,
            max_total_tokens=args.max_total_tokens,
            split_sizes=split_sizes,
            seed=args.seed,
            show_progress=(not args.no_progress),
        )

        # Print stats
        print(f"[{ds_name}:{short}] base_ok total: {stats['base_total']} (A={stats['base_A']}, B={stats['base_B']})")
        print(f"[{ds_name}:{short}] len_ok  total: {stats['len_total']}  (A={stats['len_A']}, B={stats['len_B']})")
        print(f"[{ds_name}:{short}] pool A/B: {stats['pool_A']}/{stats['pool_B']} (desired A/B: {stats['desired_need_A']}/{stats['desired_need_B']})")

        for k in split_sizes.keys():
            print(f"[{ds_name}:{short}] SPLIT {k:<7}={stats['splits'][k]} (A/B={stats['splits_A'][k]}/{stats['splits_B'][k]})")

        # Save
        output_dir = Path(args.out_prefix, short)
        output_dir.mkdir(parents=True, exist_ok=True)

        name_to_file = {
            "train": "train.jsonl",
            "val": "val.jsonl",
            "test": "test.jsonl",
            "dpo_1k": "dpo_1k.jsonl",
            "dpo_20k": "dpo_20k.jsonl",
        }

        print("=== Writing ===")
        for split_name, rows in splits.items():
            filename = name_to_file.get(split_name, f"{split_name}.jsonl")
            out_path = output_dir / filename
            save_jsonl(out_path, rows, split_name)
            print(f"wrote: {out_path}")

        if args.dpo and not allow_dpo:
            print(f"NOTE: {short} job has allow_dpo=False, so no DPO files were generated.\n")
        else:
            print("")

    print("=== Done ===")


if __name__ == "__main__":
    main()
