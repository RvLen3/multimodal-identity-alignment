import argparse
from pathlib import Path

from PIL import Image, ImageFile

# Allow loading incomplete images instead of crashing mid-run.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_inplace(img_path: Path, max_side: int, quality: int) -> tuple[bool, tuple[int, int], tuple[int, int]]:
    with Image.open(img_path) as img:
        original_size = img.size
        width, height = original_size
        longest = max(width, height)

        if longest <= max_side:
            return False, original_size, original_size

        scale = max_side / float(longest)
        new_size = (int(width * scale), int(height * scale))

        resized = img.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)
        ext = img_path.suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            resized.save(img_path, format="JPEG", quality=quality, optimize=True)
        elif ext == ".png":
            resized.save(img_path, format="PNG", optimize=True)
        else:
            # Fallback to JPEG for uncommon formats.
            new_path = img_path.with_suffix(".jpg")
            resized.save(new_path, format="JPEG", quality=quality, optimize=True)
            if new_path != img_path and img_path.exists():
                img_path.unlink()

        return True, original_size, new_size


def collect_top_photos(data_dir: Path) -> list[Path]:
    candidates = []
    for pattern in ["*/*/top_photo.jpg", "*/*/top_photo.jpeg", "*/*/top_photo.png"]:
        candidates.extend(data_dir.glob(pattern))
    return sorted(set(candidates))


def collect_all_images(data_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in data_dir.rglob("*") if p.suffix.lower() in exts])


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize all top_photo images in-place.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--max_side", type=int, default=1024, help="Max value for longest image side")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality used for jpg/jpeg outputs")
    parser.add_argument(
        "--scope",
        type=str,
        default="top_photo",
        choices=["top_photo", "all"],
        help="Choose whether to resize only top_photo or all image files under data_dir",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    if args.scope == "all":
        photo_paths = collect_all_images(data_dir)
        print(f"[Info] found {len(photo_paths)} images under {data_dir}")
    else:
        photo_paths = collect_top_photos(data_dir)
        print(f"[Info] found {len(photo_paths)} top_photo files under {data_dir}")

    resized_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, path in enumerate(photo_paths, start=1):
        try:
            changed, old_size, new_size = resize_inplace(path, args.max_side, args.quality)
            if changed:
                resized_count += 1
                print(f"[{idx}/{len(photo_paths)}] resized {path} {old_size} -> {new_size}")
            else:
                skipped_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"[{idx}/{len(photo_paths)}] failed  {path} error={exc}")

    print(
        f"[Done] resized={resized_count}, skipped={skipped_count}, failed={failed_count}, total={len(photo_paths)}"
    )


if __name__ == "__main__":
    main()
