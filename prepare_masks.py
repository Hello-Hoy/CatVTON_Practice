import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm.auto import tqdm

from catvton_runtime import build_agnostic_mask, parse_categories


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare cached agnostic masks for DressCode.")
    parser.add_argument("--data_root_path", type=str, default="data/DressCode")
    parser.add_argument("--categories", type=str, default="upper_body,lower_body,dresses")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val"])
    parser.add_argument("--max_pairs_per_category", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def iter_pair_files(category_dir: Path, split: str):
    if split == "train":
        return [category_dir / "train_pairs.txt"]
    if split == "val":
        return [category_dir / "test_pairs_paired.txt"]
    return [category_dir / "train_pairs.txt", category_dir / "test_pairs_paired.txt", category_dir / "test_pairs_unpaired.txt"]


def iter_tasks(category_dir: Path, category: str, split: str, max_pairs_per_category: int | None):
    processed = 0
    seen = set()
    for pair_file in iter_pair_files(category_dir, split):
        with open(pair_file, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            person_name = parts[0]
            if person_name in seen:
                continue
            seen.add(person_name)
            label_map_path = category_dir / "label_maps" / person_name.replace("_0.jpg", "_4.png")
            mask_path = category_dir / "agnostic_masks" / person_name.replace(".jpg", ".png")
            yield label_map_path, mask_path, category
            processed += 1
            if max_pairs_per_category is not None and processed >= max_pairs_per_category:
                return


def run_task(task, force: bool):
    label_map_path, mask_path, category = task
    build_agnostic_mask(label_map_path, mask_path, category, force=force)
    return mask_path


def main():
    args = parse_args()
    data_root = Path(args.data_root_path)
    categories = parse_categories(args.categories)

    for category in categories:
        category_dir = data_root / category
        tasks = list(iter_tasks(category_dir, category, args.split, args.max_pairs_per_category))
        if args.num_workers <= 1:
            for task in tqdm(tasks, desc=category):
                run_task(task, args.force)
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                iterator = executor.map(run_task, tasks, [args.force] * len(tasks), chunksize=32)
                for _ in tqdm(iterator, total=len(tasks), desc=category):
                    pass
        processed = len(tasks)
        print(f"{category}: prepared {processed} masks")


if __name__ == "__main__":
    main()
