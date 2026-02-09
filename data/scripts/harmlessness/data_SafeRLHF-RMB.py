import os
import json
import random
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

def seed_everything(seed=42):
    # 1. Standard Python random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def _get_last_user_content(conv_input):
    if not conv_input or not isinstance(conv_input, list):
        return ""
    # assume the last is by user
    last_msg = conv_input[-1]
    if isinstance(last_msg, dict):
        return last_msg.get("content", "")
    return str(last_msg)

def filter_by_token_length(tokenizer, prompt, res_a, res_b, max_length=1024):
    full_text = f"{prompt} {res_a} {res_b}"
    tokens = tokenizer.encode(full_text, add_special_tokens=False, truncation=False)
    return len(tokens) <= max_length

def load_RMB_all(root_dir):
    """Recursively read all files in RMB folders"""
    root = Path(root_dir)
    parsed_records = []
    files = list(root.rglob("*.json")) + list(root.rglob("*.jsonl"))
    
    print(f"Reading {len(files)} files from {root_dir}...")

    for p in files:
        try:
            with p.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content: continue
                data = json.loads(content) if content.startswith('[') else [json.loads(l) for l in content.splitlines() if l.strip()]
                
                for rec in data:
                    try:
                        # 1. extract Prompt (from conversation_input )
                        prompt = _get_last_user_content(rec.get("conversation_input", []))
                        # 2. extract Chosen/Rejected 
                        chosen = rec.get("chosen", {}).get("answer", "")
                        rejected = rec.get("reject", {}).get("answer", "")

                        if prompt.strip() and chosen.strip() and rejected.strip():
                            parsed_records.append({
                                "prompt": prompt,
                                "safe": chosen,
                                "harm": rejected
                            })
                    except:
                        continue
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
            
    return parsed_records

def prepare_and_save(tokenizer, data_list, output_path, n_train=1000, n_val=1000, n_test=1000, is_RMB=False, max_total_tokens=1024):
    os.makedirs(output_path, exist_ok=True)

    random.shuffle(data_list)
    # filter long data
    valid_data = []
    for row in data_list:
        if is_RMB:
            p, s, h = row["prompt"], row["safe"], row["harm"]
        else:
            p = row["prompt"]
            s = row["response_0"] if row["safer_response_id"] == 0 else row["response_1"]
            h = row["response_1"] if row["safer_response_id"] == 0 else row["response_0"]
        
        if filter_by_token_length(tokenizer, p, s, h, max_length=max_total_tokens):
            valid_data.append((p, s, h))

    splits = {"train.jsonl": (0, n_train), "val.jsonl": (n_train, n_train + n_val), "test.jsonl": (n_train + n_val, n_train + n_val + n_test)}
    print("Data splits: ", splits)
    print(f"\nProcessing {output_path} (Valid records: {len(valid_data)})...")
    for filename, (start, end) in splits.items():
        if start >= len(valid_data):
            print(f" ! Warning: {filename} skipped (index {start} out of range).")
            continue
            
        subset = valid_data[start:min(end, len(valid_data))]
        save_file = os.path.join(output_path, filename)
        
        with open(save_file, "w", encoding="utf-8") as f:
            for p, s, h in subset:
                # random flip
                if random.random() > 0.5:
                    record = {"instruction": p, "response_a": s, "response_b": h, "ground_truth": "A"}
                else:
                    record = {"instruction": p, "response_a": h, "response_b": s, "ground_truth": "B"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f" - Saved {len(subset)} rows to {save_file}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name/path used to count tokens.")
    ap.add_argument("--max-total-tokens", type=int, default=1024, help="Max tokens allowed for concatenated instruction+resp1+resp2.")

    ap.add_argument("--bench_domain", type=str, default="RMB", help="Benchmark domain.")
    ap.add_argument("--target_domain", type=str, default="SafeRLHF", help="Target domain.")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="./data/harmlessness", help="Output file folder.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = ap.parse_args()

    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=999999)
    base_save_path = f"./harmlessness/{args.bench_domain}-{args.target_domain}"
    
    # Load RMB-Harmlessness
    print(f"Loading RMB-Harmlessness...")
    RMB_raw_path = "./raw_data/RMB-Harmlessness"
    RMB_data = load_RMB_all(RMB_raw_path)
    print(f"Loaded {len(RMB_data)} rows from RMB-Harmlessness")
  
    # Load PKU-SafeRLHF
    print("\nLoading PKU-SafeRLHF...")
    safe_rlhf = load_dataset("PKU-Alignment/PKU-SafeRLHF")['train']
    SafeRLHF_data = list(safe_rlhf)  
    print(f"Loaded {len(SafeRLHF_data)} rows from PKU-SafeRLHF")
    
    if args.bench_domain == "RMB" and args.target_domain == "SafeRLHF":
        prepare_and_save(tokenizer, RMB_data, os.path.join(base_save_path, f"{args.bench_domain}-{args.target_domain}-Bench"), is_RMB=True, max_total_tokens=args.max_total_tokens, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test)
        prepare_and_save(tokenizer, SafeRLHF_data, os.path.join(base_save_path, f"{args.bench_domain}-{args.target_domain}-Target"), is_RMB=False, max_total_tokens=args.max_total_tokens, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test)
        print(f"Saved {len(RMB_data)} rows to {os.path.join(base_save_path, f'{args.bench_domain}-{args.target_domain}-Bench')}")
        print(f"Saved {len(SafeRLHF_data)} rows to {os.path.join(base_save_path, f'{args.bench_domain}-{args.target_domain}-Target')}")

    elif args.bench_domain == "SafeRLHF" and args.target_domain == "RMB":
        prepare_and_save(tokenizer, RMB_data, os.path.join(base_save_path, f"{args.bench_domain}-{args.target_domain}-Target"), is_RMB=True, max_total_tokens=args.max_total_tokens, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test)
        prepare_and_save(tokenizer, SafeRLHF_data, os.path.join(base_save_path, f"{args.bench_domain}-{args.target_domain}-Bench"), is_RMB=False, max_total_tokens=args.max_total_tokens, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test)
        print(f"Saved {len(SafeRLHF_data)} rows to {os.path.join(base_save_path, f'{args.bench_domain}-{args.target_domain}-Bench')}")
        print(f"Saved {len(RMB_data)} rows to {os.path.join(base_save_path, f'{args.bench_domain}-{args.target_domain}-Target')}")

    else:
        raise ValueError(f"Invalid benchmark domain: {args.bench_domain}")

  
if __name__ == "__main__":
    main()