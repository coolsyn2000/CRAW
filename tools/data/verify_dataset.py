from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_files(root: Path, patterns: tuple[str, ...]) -> int:
    return sum(sum(1 for _ in root.rglob(pattern)) for pattern in patterns)


def verify_sidd(data_root: Path) -> bool:
    sidd_root = data_root / "SIDD"
    prepared = sidd_root / "nafnet_raw_to_linear_spatial_rggb"
    valid = True
    for split in ("train", "val"):
        metadata_path = prepared / f"{split}.lmdb" / "meta_info.json"
        if not metadata_path.is_file():
            print(f"missing: {metadata_path}")
            valid = False
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        entries = int(metadata.get("entry_count", len(metadata.get("entries", []))))
        print(
            f"ok: SIDD {split} LMDB entries={entries} "
            f"packed_size={metadata.get('packed_size')} format={metadata.get('format_version')}"
        )
        valid &= entries > 0
    raw_root = sidd_root / "medium_raw"
    mat_count = count_files(raw_root, ("*_NOISY_RAW_*.MAT", "*_NOISY_RAW_*.mat"))
    print(f"info: SIDD source noisy RAW files={mat_count}")
    return valid


def verify_lsdir(data_root: Path) -> bool:
    image_root = data_root / "LSDIR" / "HR" / "train"
    count = count_files(image_root, ("*.png", "*.jpg", "*.jpeg"))
    print(f"{'ok' if count else 'missing'}: LSDIR HR training images={count} root={image_root}")
    return count > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the public SIDD/LSDIR layout.")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    if not verify_sidd(root) or not verify_lsdir(root):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
