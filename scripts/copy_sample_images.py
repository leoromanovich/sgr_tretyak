from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_source_image(source_dir: Path, note_id: str) -> Path | None:
    candidates = [
        source_dir / f"{note_id}.jpg",
        source_dir / f"{note_id}.jpeg",
        source_dir / f"{note_id}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def copy_images(pages_dir: Path, source_dir: Path, dest_dir: Path) -> tuple[int, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0

    for note_path in sorted(pages_dir.glob("*.md")):
        note_id = note_path.stem
        src_image = find_source_image(source_dir, note_id)
        if not src_image:
            missing += 1
            continue
        dst_path = dest_dir / src_image.name
        shutil.copy2(src_image, dst_path)
        copied += 1

    return copied, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy note preview images for the sample Obsidian export.",
    )
    parser.add_argument(
        "--pages-dir",
        required=True,
        type=Path,
        help="Directory with markdown notes (used to derive note IDs).",
    )
    parser.add_argument(
        "--source-images-dir",
        required=True,
        type=Path,
        help="Directory with original images (e.g. data/obsidian/images).",
    )
    parser.add_argument(
        "--dest-images-dir",
        required=True,
        type=Path,
        help="Destination directory for copied images (will be created).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages_dir: Path = args.pages_dir
    source_dir: Path = args.source_images_dir
    dest_dir: Path = args.dest_images_dir

    if not pages_dir.exists():
        raise SystemExit(f"Pages directory not found: {pages_dir}")
    if not source_dir.exists():
        raise SystemExit(f"Source images directory not found: {source_dir}")

    copied, missing = copy_images(pages_dir, source_dir, dest_dir)
    print(
        f"Copied {copied} image(s) to {dest_dir} "
        f"(missing for {missing} note(s))."
    )


if __name__ == "__main__":
    main()
