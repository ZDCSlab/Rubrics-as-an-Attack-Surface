"""
- Ultra-Real: bench: ultrafeedback; target: real_world.
- Ultra-Creative: bench: ultrafeedback; target: problem_solving.

Output structure:
  helpfulness/Ultra-Real/
    Bench/        <- ultrafeedback (UltraFeedback)
    Target/       <- real_world (ChatbotArena)

  helpfulness/Ultra-Creative/
    Bench/        <- ultrafeedback (UltraFeedback)
    Target/       <- problem_solving (ChatbotArena)
"""

import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser(description="Reorganize helpfulness into harmlessness-style layout.")
    ap.add_argument("--raw_data_dir", type=str, default="./raw_data",
                    help="Path to data/helpfulness (contains problem_solving/, real_world/, ultrafeedback/).")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Parent output dir (default: same as raw_data-dir).")
    args = ap.parse_args()

    out_base = (os.path.normpath(args.out_dir) if args.out_dir else args.raw_data_dir).rstrip("/")
    helpfulness_base = os.path.normpath(args.raw_data_dir).rstrip("/")

    # (dataset_name, bench_source, target_source)
    datasets = [
        ("Ultra-Real", "ultrafeedback", "real_world"),
        ("Ultra-Creative", "ultrafeedback", "problem_solving"),
    ]

    for dataset_name, bench_source, target_source in datasets:
        for role, source_subdir in (("Bench", bench_source), ("Target", target_source)):
            src_dir = os.path.join(helpfulness_base, source_subdir)
            dst_dir = os.path.join(out_base, dataset_name, f"{dataset_name}-{role}")
            if not os.path.isdir(src_dir):
                print(f"Skip (missing dir): {src_dir}")
                continue
            os.makedirs(dst_dir, exist_ok=True)
            for split in ("train", "val", "test"):
                src_file = os.path.join(src_dir, f"{split}.jsonl")
                dst_file = os.path.join(dst_dir, f"{split}.jsonl")
                if not os.path.isfile(src_file):
                    print(f"Skip (missing file): {src_file}")
                    continue
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")


if __name__ == "__main__":
    main()
