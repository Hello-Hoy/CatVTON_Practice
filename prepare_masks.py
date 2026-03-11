import argparse
from pathlib import Path

from tqdm.auto import tqdm

from catvton_runtime import build_agnostic_mask, parse_categories


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare cached agnostic masks for DressCode.")
    parser.add_argument("--data_root_path", type=str, default="data/DressCode")
    parser.add_argument("--categories", type=str, default="upper_body,lower_body,dresses")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val"])
    parser.add_argument("--max_pairs_per_category", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def iter_pair_files(category_dir: Path, split: str):
    if split == "train":
        return [category_dir / "train_pairs.txt"]
    if split == "val":
        return [category_dir / "test_pairs_paired.txt"]
    return [category_dir / "train_pairs.txt", category_dir / "test_pairs_paired.txt", category_dir / "test_pairs_unpaired.txt"]


def main():
    args = parse_args()
    data_root = Path(args.data_root_path)
    categories = parse_categories(args.categories)

    for category in categories:
        category_dir = data_root / category
        processed = 0
        seen = set()
        for pair_file in iter_pair_files(category_dir, args.split):
            with open(pair_file, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
            for line in tqdm(lines, desc=f"{category}:{pair_file.name}"):
                parts = line.strip().split()
                if not parts:
                    continue
                person_name = parts[0]
                if person_name in seen:
                    continue
                seen.add(person_name)
                label_map_path = category_dir / "label_maps" / person_name.replace("_0.jpg", "_4.png")
                mask_path = category_dir / "agnostic_masks" / person_name.replace(".jpg", ".png")
                build_agnostic_mask(label_map_path, mask_path, category, force=args.force)
                processed += 1
                if args.max_pairs_per_category is not None and processed >= args.max_pairs_per_category:
                    break
            if args.max_pairs_per_category is not None and processed >= args.max_pairs_per_category:
                break
        print(f"{category}: prepared {processed} masks")


if __name__ == "__main__":
    main()
