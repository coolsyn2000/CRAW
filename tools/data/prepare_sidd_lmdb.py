from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import lmdb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from raw_codec.data.sidd_raw_dataset import (
    SiddRawDataset,
    _canonicalize_packed_raw,
    _load_mat_image,
)
from raw_codec.data.sidd_color import load_sidd_render_profile_for_scene


def crop_positions(length: int, crop_size: int, step: int) -> list[int]:
    if crop_size > length:
        raise ValueError(f"Cannot crop {crop_size} pixels from length {length}.")
    positions = list(range(0, length - crop_size + 1, step))
    final = length - crop_size
    if positions[-1] != final:
        positions.append(final)
    return positions


def choose_validation_scenes(
    pairs: list[Any],
    count: int,
    seed: int,
) -> set[str]:
    scenes_by_camera: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        if pair.scene_id not in scenes_by_camera[pair.camera]:
            scenes_by_camera[pair.camera].append(pair.scene_id)
    total_scenes = sum(len(scenes) for scenes in scenes_by_camera.values())
    if not 0 < count < total_scenes:
        raise ValueError(f"val_scene_count must be in [1,{total_scenes - 1}].")

    rng = random.Random(seed)
    quotas: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    assigned = 0
    for camera, scenes in sorted(scenes_by_camera.items()):
        rng.shuffle(scenes)
        exact = count * len(scenes) / total_scenes
        quota = int(exact)
        quotas[camera] = quota
        assigned += quota
        fractions.append((exact - quota, camera))
    for _, camera in sorted(fractions, reverse=True)[: count - assigned]:
        quotas[camera] += 1

    selected: set[str] = set()
    for camera, scenes in sorted(scenes_by_camera.items()):
        selected.update(scenes[: quotas[camera]])
    if len(selected) != count:
        raise RuntimeError(f"Expected {count} validation scenes, selected {len(selected)}.")
    return selected


def quantize_raw(signal: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(signal, 0.0, 1.0) * 65535.0).astype(np.uint16)


def write_entry(
    transaction: lmdb.Transaction,
    index: int,
    noisy: np.ndarray,
    clean: np.ndarray,
) -> None:
    transaction.put(f"{index:08d}/noisy".encode("ascii"), noisy.tobytes(order="C"))
    transaction.put(f"{index:08d}/clean".encode("ascii"), clean.tobytes(order="C"))


def build_split(
    output_path: Path,
    pairs: list[Any],
    packed_size: int,
    step: int,
    validation_patches_per_pair: int,
    seed: int,
    map_size_gb: int,
) -> None:
    output_path.mkdir(parents=True, exist_ok=False)
    environment = lmdb.open(
        str(output_path),
        map_size=int(map_size_gb) * 1024**3,
        subdir=True,
        meminit=False,
        map_async=True,
    )
    entries: list[dict[str, Any]] = []
    render_profiles: dict[str, dict[str, Any]] = {}
    transaction = environment.begin(write=True)
    entry_index = 0
    is_validation = output_path.name.startswith("val")

    try:
        for pair_index, pair in enumerate(pairs):
            if pair.scene_id not in render_profiles:
                render_profiles[pair.scene_id] = load_sidd_render_profile_for_scene(
                    pair.noisy_path,
                    pair.scene_id,
                )
            noisy = _canonicalize_packed_raw(
                _load_mat_image(pair.noisy_path),
                pair.bayer_pattern,
            ).numpy()
            clean = _canonicalize_packed_raw(
                _load_mat_image(pair.gt_path),
                pair.bayer_pattern,
            ).numpy()
            if noisy.shape != clean.shape:
                raise ValueError(f"Shape mismatch for {pair.noisy_path}: {noisy.shape} vs {clean.shape}.")
            _, height, width = noisy.shape

            if is_validation:
                generator = np.random.default_rng(seed + pair_index)
                positions = [
                    (
                        int(generator.integers(0, height - packed_size + 1)),
                        int(generator.integers(0, width - packed_size + 1)),
                    )
                    for _ in range(validation_patches_per_pair)
                ]
            else:
                positions = [
                    (top, left)
                    for top in crop_positions(height, packed_size, step)
                    for left in crop_positions(width, packed_size, step)
                ]

            for top, left in positions:
                crop = np.s_[..., top : top + packed_size, left : left + packed_size]
                noisy_crop = quantize_raw(noisy[crop])
                clean_crop = quantize_raw(clean[crop])
                write_entry(transaction, entry_index, noisy_crop, clean_crop)
                entries.append(
                    {
                        "scene": pair.scene_id,
                        "camera": pair.camera,
                        "iso": pair.iso,
                        "source_bayer_pattern": pair.bayer_pattern,
                        "top": top,
                        "left": left,
                        "noisy_path": str(pair.noisy_path),
                        "clean_path": str(pair.gt_path),
                    }
                )
                entry_index += 1
                if entry_index % 128 == 0:
                    transaction.commit()
                    transaction = environment.begin(write=True)
            print(
                f"split={output_path.stem} pair={pair_index + 1}/{len(pairs)} "
                f"entries={entry_index}",
                flush=True,
            )
        transaction.commit()
        environment.sync()
    finally:
        environment.close()

    metadata = {
        "format_version": 3,
        "cfa_canonicalization": "whole_image_spatial_transform_to_rggb",
        "dtype": "uint16",
        "channels": 4,
        "packed_size": packed_size,
        "entry_count": len(entries),
        "render_profiles": render_profiles,
        "entries": entries,
    }
    with (output_path / "meta_info.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare canonical packed SIDD RAW pairs for NAFNet training."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--output-root",
        required=True,
    )
    parser.add_argument("--train-packed-size", type=int, default=256)
    parser.add_argument("--train-step", type=int, default=192)
    parser.add_argument("--val-packed-size", type=int, default=128)
    parser.add_argument("--val-patches-per-pair", type=int, default=32)
    parser.add_argument("--val-scene-count", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=10)
    parser.add_argument("--map-size-gb", type=int, default=80)
    parser.add_argument("--bayer-patterns-csv", default=None)
    parser.add_argument("--noise-level-functions-csv", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    source_root = Path(args.source_root).expanduser()
    bayer_csv = args.bayer_patterns_csv
    nlf_csv = args.noise_level_functions_csv
    if bayer_csv is None and (source_root.parent / "bayer_patterns.csv").is_file():
        bayer_csv = str(source_root.parent / "bayer_patterns.csv")
    if nlf_csv is None and (source_root.parent / "noise_level_functions.csv").is_file():
        nlf_csv = str(source_root.parent / "noise_level_functions.csv")
    source = SiddRawDataset(
        root=source_root,
        patch_size=1,
        length=0,
        deterministic=True,
        augment=False,
        bayer_patterns_csv=bayer_csv,
        noise_level_functions_csv=nlf_csv,
    )
    validation_scenes = choose_validation_scenes(
        source.pairs,
        count=args.val_scene_count,
        seed=args.split_seed,
    )
    train_pairs = [pair for pair in source.pairs if pair.scene_id not in validation_scenes]
    val_pairs = [pair for pair in source.pairs if pair.scene_id in validation_scenes]
    split_manifest = {
        "seed": args.split_seed,
        "train_scene_count": len({pair.scene_id for pair in train_pairs}),
        "val_scene_count": len(validation_scenes),
        "train_pair_count": len(train_pairs),
        "val_pair_count": len(val_pairs),
        "validation_scenes": sorted(validation_scenes),
    }
    with (output_root / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2)

    build_split(
        output_root / "train.lmdb",
        train_pairs,
        packed_size=args.train_packed_size,
        step=args.train_step,
        validation_patches_per_pair=args.val_patches_per_pair,
        seed=args.split_seed,
        map_size_gb=args.map_size_gb,
    )
    build_split(
        output_root / "val.lmdb",
        val_pairs,
        packed_size=args.val_packed_size,
        step=args.train_step,
        validation_patches_per_pair=args.val_patches_per_pair,
        seed=args.split_seed,
        map_size_gb=args.map_size_gb,
    )
    print(json.dumps(split_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
