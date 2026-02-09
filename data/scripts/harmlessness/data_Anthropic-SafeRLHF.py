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

def filter_by_token_length(tokenizer, prompt, res_a, res_b, max_length=1024):
    full_text = f"{prompt} {res_a} {res_b}"
    tokens = tokenizer.encode(full_text, add_special_tokens=False, truncation=False)
    return len(tokens) <= max_length

def parse_hh_rlhf_example(example):
    """
    Parses Anthropic HH-RLHF format.
    Ensures safe_content/harm_content only contain the final assistant response.
    """
    def extract_last_assistant_response(text):
        # 1. Split at the last "Assistant: " marker
        parts = text.rsplit("\n\nAssistant: ", 1)
        if len(parts) < 2:
            return "", ""
        
        prompt_history = parts[0].strip()
        last_response = parts[1].strip()
        
        # 2. Safety check: If the response contains a subsequent "Human:" turn, 
        # (which can happen in some multi-turn versions), strip it.
        if "Human:" in last_response:
            last_response = last_response.split("Human:")[0].strip()
            
        return prompt_history, last_response

    # Extract safe content (chosen)
    prompt, safe_content = extract_last_assistant_response(example["chosen"])
    
    # Extract harm content (rejected)
    # Note: We use the prompt derived from the 'chosen' path as the standard 
    # instruction, as the history prefix is identical.
    _, harm_content = extract_last_assistant_response(example["rejected"])

    return {
        "prompt": prompt,
        "safe_content": safe_content,
        "harm_content": harm_content
    }

def prepare_and_save(tokenizer, data_iterable, output_path, n_train=1000, n_val=1000, n_test=1000, is_hh=False, is_dpo=False, max_total_tokens=1024):
    """
    Unified function that filters by length and saves datasets.
    Iterates until target counts are met for each split.
    """
    os.makedirs(output_path, exist_ok=True)

    # Target counts for Train, Val, and Test
    splits = {
        "train.jsonl": n_train,
        "val.jsonl": n_val,
        "test.jsonl": n_test
    }
    if is_dpo:
        splits["dpo_20k.jsonl"] = 20000
        splits["dpo_1k.jsonl"] = 1000

    data_iter = iter(data_iterable)

    print(f"\nProcessing {output_path}...")
    for filename, n_samples in splits.items():
        save_file = os.path.join(output_path, filename)
        count = 0
        
        with open(save_file, "w", encoding="utf-8") as f:
            while count < n_samples:
                try:
                    row = next(data_iter)
                except StopIteration:
                    print(f" ! Warning: Ran out of data for {filename}")
                    break

                # --- 1. Parsing Logic ---
                if is_hh:
                    parsed = parse_hh_rlhf_example(row)
                    prompt = parsed["prompt"]
                    safe_content = parsed["safe_content"]
                    harm_content = parsed["harm_content"]
                else:
                    # SafeRLHF Logic
                    prompt = row["prompt"]
                    if row["safer_response_id"] == 0:
                        safe_content, harm_content = row["response_0"], row["response_1"]
                    else:
                        safe_content, harm_content = row["response_1"], row["response_0"]

                # --- 2. Length Filtering & Validation ---
                if not (prompt and safe_content and harm_content):
                    continue
                
                if not filter_by_token_length(tokenizer, prompt, safe_content, harm_content, max_length=max_total_tokens):
                    continue

                # --- 3. Positional Bias Randomization ---
                if random.random() > 0.5:
                    record = {
                        "instruction": prompt,
                        "response_a": safe_content,
                        "response_b": harm_content,
                        "ground_truth": "A"
                    }
                else:
                    record = {
                        "instruction": prompt,
                        "response_a": harm_content,
                        "response_b": safe_content,
                        "ground_truth": "B"
                    }
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                
        print(f" - Saved {count} rows to {save_file}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name/path used to count tokens.")
    ap.add_argument("--max-total-tokens", type=int, default=1024, help="Max tokens allowed for concatenated instruction+resp1+resp2.")

    ap.add_argument("--bench_domain", type=str, default="Anthropic", help="Benchmark domain.")
    ap.add_argument("--target_domain", type=str, default="SafeRLHF", help="Target domain.")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="./data/harmlessness", help="Output file folder.")
    ap.add_argument("--is_dpo", action="store_true", help="Whether to save DPO data.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = ap.parse_args()

    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=999999)
    base_save_path = f"./harmlessness/{args.bench_domain}-{args.target_domain}"

    # 1. Load Datasets
    print("\nLoading PKU-SafeRLHF...")
    safe_rlhf = load_dataset("PKU-Alignment/PKU-SafeRLHF")['train']
    hh_rlhf = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")['train']

    # 2. Shuffle
    safe_rlhf = safe_rlhf.shuffle(seed=args.seed)
    hh_rlhf = hh_rlhf.shuffle(seed=args.seed)
    print(f"SafeRLHF length: {len(safe_rlhf)}")
    print(f"HH-RLHF length: {len(hh_rlhf)}")

    # 3. Process and Save
    if args.bench_domain == "Anthropic" and args.target_domain == "SafeRLHF":
        prepare_and_save(tokenizer, safe_rlhf, os.path.join(base_save_path, f"{args.bench_domain}-{args.target_domain}-Target"), is_hh=False, is_dpo=args.is_dpo, max_total_tokens=args.max_total_tokens, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test)
        prepare_and_save(tokenizer, hh_rlhf, os.path.join(base_save_path, f"{args.bench_domain}-{args.target_domain}-Bench"), is_hh=True,  is_dpo=args.is_dpo, max_total_tokens=args.max_total_tokens, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test)
        print(f"Saved {len(safe_rlhf)} rows to {os.path.join(base_save_path, f'{args.bench_domain}-{args.target_domain}-Target')}")
        print(f"Saved {len(hh_rlhf)} rows to {os.path.join(base_save_path, f'{args.bench_domain}-{args.target_domain}-Bench')}")
    elif args.bench_domain == "SafeRLHF" and args.target_domain == "Anthropic":
        prepare_and_save(tokenizer, safe_rlhf, os.path.join(base_save_path, f"{args.bench_domain}-{args.target_domain}-Bench"), is_hh=False, is_dpo=args.is_dpo, max_total_tokens=args.max_total_tokens, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test)
        prepare_and_save(tokenizer, hh_rlhf, os.path.join(base_save_path, f"{args.bench_domain}-{args.target_domain}-Target"), is_hh=True, is_dpo=args.is_dpo,max_total_tokens=args.max_total_tokens, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test)
        print(f"Saved {len(safe_rlhf)} rows to {os.path.join(base_save_path, f'{args.bench_domain}-{args.target_domain}-Bench')}")
        print(f"Saved {len(hh_rlhf)} rows to {os.path.join(base_save_path, f'{args.bench_domain}-{args.target_domain}-Target')}")
    else:
        raise ValueError(f"Invalid benchmark domain: {args.bench_domain}")
        
if __name__ == "__main__":
    main()