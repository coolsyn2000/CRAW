from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch.utils.data import DataLoader, Dataset

from raw_codec.data import (
    FixedRatioMixedBatchSampler,
    MixedRawDataset,
    SiddNafnetRawDataset,
    SyntheticRawDataset,
    resolve_cfa_pattern,
    resolve_noise_config,
    resolve_unprocess_config,
)


_LOADER_KEYS = {
    "batch_size",
    "batch_source_counts",
    "num_workers",
    "prefetch_factor",
    "pin_memory",
}


def _linear_rgb_has_black_level(config: dict[str, Any]) -> bool:
    target = dict(config.get("target", {}))
    return bool(
        target.get(
            "include_black_level",
            target.get(
                "linear_rgb_include_black_level",
                target.get("linear_rgb_has_black_level", False),
            ),
        )
    )


def _dataset_from_document(config: dict[str, Any], document: dict[str, Any], split: str) -> Dataset:
    cfg = deepcopy(document)
    mode = str(cfg.pop("mode", "image")).lower().replace("-", "_")
    for key in _LOADER_KEYS:
        cfg.pop(key, None)

    if mode == "mixed":
        source_documents = list(cfg.pop("datasets"))
        datasets = [_dataset_from_document(config, source, split) for source in source_documents]
        names = [str(source.get("name", f"source_{index}")) for index, source in enumerate(source_documents)]
        for source in source_documents:
            source.pop("name", None)
        return MixedRawDataset(
            datasets=datasets,
            names=names,
            weights=cfg.pop("weights", None),
            **cfg,
        )

    cfg.pop("name", None)
    if mode in {"sidd_pretiled_raw", "sidd_lmdb"}:
        if _linear_rgb_has_black_level(config):
            cfg["include_black_level"] = True
        return SiddNafnetRawDataset(split=split, **cfg)

    if mode in {"image", "synthetic", "random"}:
        cfg["mode"] = "random" if mode in {"synthetic", "random"} else "image"
        unprocess_cfg = resolve_unprocess_config(config)
        if _linear_rgb_has_black_level(config):
            unprocess_cfg["linear_rgb_include_black_level"] = True
        return SyntheticRawDataset(
            **cfg,
            noise_cfg=resolve_noise_config(config),
            unprocess_cfg=unprocess_cfg,
            cfa_pattern=resolve_cfa_pattern(config),
        )

    raise ValueError(
        f"Unsupported data mode {mode!r}. Public release supports mixed, "
        "sidd_lmdb, image, and random."
    )


def build_dataset(config: dict[str, Any], split: str) -> Dataset:
    data_cfg = dict(config.get("data", {}))
    if split not in data_cfg:
        raise KeyError(f"Config has no data.{split} section.")
    return _dataset_from_document(config, data_cfg[split], split)


def build_loader(config: dict[str, Any], split: str, *, device_type: str) -> DataLoader:
    split_cfg = dict(config["data"][split])
    dataset = build_dataset(config, split)
    batch_size = int(split_cfg.get("batch_size", 1))
    workers = int(split_cfg.get("num_workers", 0))
    common: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(split_cfg.get("pin_memory", device_type == "cuda")),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        common["prefetch_factor"] = int(split_cfg.get("prefetch_factor", 2))

    source_counts = split_cfg.get("batch_source_counts")
    if split == "train" and isinstance(dataset, MixedRawDataset) and source_counts is not None:
        sampler = FixedRatioMixedBatchSampler(
            dataset,
            source_counts=source_counts,
            global_batch_size=batch_size,
            per_rank_batch_size=batch_size,
            num_batches=max(int(split_cfg.get("length", len(dataset))) // batch_size, 1),
            seed=int(config.get("train", {}).get("seed", 0)),
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        drop_last=split == "train",
        **common,
    )


def build_validation_loaders(config: dict[str, Any], *, device_type: str) -> dict[str, DataLoader]:
    validation_sets = dict(config.get("data", {}).get("validation_sets", {}))
    if not validation_sets:
        return {"val": build_loader(config, "val", device_type=device_type)}
    loaders: dict[str, DataLoader] = {}
    for name, document in validation_sets.items():
        effective = deepcopy(config)
        effective.setdefault("data", {})["val"] = deepcopy(document)
        loaders[str(name)] = build_loader(effective, "val", device_type=device_type)
    return loaders
