#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
import shutil
from typing import List

def pick_pickles(root: Path, exts={".pkl", ".pickle"}) -> List[Path]:
    # Only pick files directly under root (not in train/val/test)
    ignore = {"train", "val", "test"}
    return sorted(
        [p for p in root.iterdir()
         if p.is_file() and p.suffix.lower() in exts and p.parent == root]
    )

def split_counts(n: int, train_ratio=0.7, val_ratio=0.15):
    train = int(n * train_ratio)
    val = int(n * val_ratio)
    test = n - train - val
    return train, val, test

def ensure_dirs(root: Path):
    for sub in ("train", "val", "test"):
        (root / sub).mkdir(exist_ok=True)

def move_files(files: List[Path], dest_dir: Path):
    for f in files:
        target = dest_dir / f.name
        # If a file with the same name exists in dest, add a numeric suffix
        if target.exists():
            stem, suf = f.stem, f.suffix
            i = 1
            while True:
                candidate = dest_dir / f"{stem}_{i}{suf}"
                if not candidate.exists():
                    target = candidate
                    break
                i += 1
        shutil.move(str(f), str(target))

def main():
    ap = argparse.ArgumentParser(
        description="Split pickle files into train/val/test (70/15/15) subdirectories."
    )
    ap.add_argument("dir", type=Path, help="Directory containing pickle files")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    ap.add_argument("--exts", type=str, default=".pkl,.pickle",
                    help="Comma-separated extensions to include (default: .pkl,.pickle)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be moved without moving")
    args = ap.parse_args()

    root = args.dir.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    exts = {e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
            for e in args.exts.split(",")}
    all_pickles = pick_pickles(root, exts=exts)

    # Exclude files already inside train/val/test (script only looks at root level by default)
    n = len(all_pickles)
    if n == 0:
        print("No pickle files found at root level. Nothing to do.")
        return

    r = random.Random(args.seed)
    r.shuffle(all_pickles)

    n_train, n_val, n_test = split_counts(n)
    train_files = all_pickles[:n_train]
    val_files = all_pickles[n_train:n_train + n_val]
    test_files = all_pickles[n_train + n_val:]

    print(f"Found {n} file(s). Splitting into:")
    print(f"  train: {len(train_files)}")
    print(f"  val:   {len(val_files)}")
    print(f"  test:  {len(test_files)}")

    if args.dry_run:
        print("\nDry run (no files moved). Preview:")
        print("  -> train:", [f.name for f in train_files])
        print("  -> val:  ", [f.name for f in val_files])
        print("  -> test: ", [f.name for f in test_files])
        return

    ensure_dirs(root)
    move_files(train_files, root / "train")
    move_files(val_files, root / "val")
    move_files(test_files, root / "test")

    print("\nDone!")

if __name__ == "__main__":
    main()

# python tool_split_dataset.py $DATA_ROOT/aria_processed_cpf_corrected