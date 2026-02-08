import argparse, os, random, shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]

def copy_many(items, dst_folder: Path):
    dst_folder.mkdir(parents=True, exist_ok=True)
    for p in items:
        shutil.copy2(p, dst_folder / p.name)

def main():
    # Getting Args from user
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with class subfolders: sit/ sleep/")
    ap.add_argument("--output", required=True, help="Output folder for splits")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Validate ratio of train/validation/test must = 1.0 (100%)
    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise SystemExit("train+val+test must equal 1.0")

    # Random seed used to swing dataset to not same for each round
    random.seed(args.seed)

    # Input directory
    in_root = Path(args.input)

    # Output directory
    out_root = Path(args.output)

    # List class label
    classes = [p.name for p in in_root.iterdir() if p.is_dir()]
    if not classes:
        raise SystemExit("No class folders found. Expected: input/sit and input/sleep")

    # Creating output directory with class label
    for split in ["train", "val", "test"]:
        for c in classes:
            (out_root / split / c).mkdir(parents=True, exist_ok=True)

    # Loop All labeled class
    for c in classes:
        print(f"Processing class {c}")

        # Getting all image into each class folder
        imgs = list_images(in_root / c)
        if not imgs:
            print(f"[WARN] no images for class: {c}")
            continue
        random.shuffle(imgs)

        # Counting total image into folder
        n = len(imgs)
        print(f"Found image of class {c} = {n} images")

        # Calculate image amount to train/validate/test from input ratio
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        
        # Getting image from calculated amount
        train_set = imgs[:n_train]
        val_set = imgs[n_train:n_train+n_val]
        test_set = imgs[n_train+n_val:]

        # Copy image to output folder
        copy_many(train_set, out_root / "train" / c)
        copy_many(val_set, out_root / "val" / c)
        copy_many(test_set, out_root / "test" / c)

        print(f"[OK] {c}: n={n} train={len(train_set)} val={len(val_set)} test={len(test_set)}")

if __name__ == "__main__":
    main()
