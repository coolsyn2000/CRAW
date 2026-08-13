from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch
from torch.utils.data import BatchSampler, Dataset


def _normalize_source_counts(
    source_counts: Mapping[str, int] | Sequence[int],
    names: Sequence[str],
) -> list[int]:
    if isinstance(source_counts, Mapping):
        unknown = sorted(set(str(key) for key in source_counts) - set(names))
        if unknown:
            raise ValueError(f"Unknown mixed source names in batch_source_counts: {unknown}. Known names: {list(names)}")
        return [int(source_counts.get(name, 0)) for name in names]
    counts = [int(value) for value in source_counts]
    if len(counts) != len(names):
        raise ValueError(f"batch_source_counts length {len(counts)} must match source count {len(names)}.")
    return counts


def _counts_from_ratios(
    ratios: Mapping[str, float] | Sequence[float],
    names: Sequence[str],
    batch_size: int,
) -> list[int]:
    if isinstance(ratios, Mapping):
        unknown = sorted(set(str(key) for key in ratios) - set(names))
        if unknown:
            raise ValueError(f"Unknown mixed source names in batch_source_ratios: {unknown}. Known names: {list(names)}")
        values = [float(ratios.get(name, 0.0)) for name in names]
    else:
        values = [float(value) for value in ratios]
        if len(values) != len(names):
            raise ValueError(f"batch_source_ratios length {len(values)} must match source count {len(names)}.")
    if any(value < 0.0 for value in values) or sum(values) <= 0.0:
        raise ValueError(f"batch_source_ratios must be non-negative and not all zero, got {values}.")

    total = float(sum(values))
    raw_counts = [value / total * int(batch_size) for value in values]
    counts = [int(value) for value in raw_counts]
    remainder = int(batch_size) - sum(counts)
    order = sorted(range(len(values)), key=lambda idx: raw_counts[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


class FixedRatioMixedBatchSampler(BatchSampler):
    """Yield mixed-source batches with exact source counts.

    Each yielded index is a ``(dataset_id, source_index)`` tuple understood by
    :class:`MixedRawDataset`. For DDP, every rank builds the same global batch
    and receives a non-overlapping slice, so the global batch composition exactly
    matches ``source_counts``.
    """

    def __init__(
        self,
        dataset: "MixedRawDataset",
        source_counts: Mapping[str, int] | Sequence[int] | None = None,
        source_ratios: Mapping[str, float] | Sequence[float] | None = None,
        global_batch_size: int = 1,
        per_rank_batch_size: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        num_batches: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if source_counts is None and source_ratios is None:
            raise ValueError("FixedRatioMixedBatchSampler requires source_counts or source_ratios.")
        if source_counts is not None and source_ratios is not None:
            raise ValueError("Use only one of source_counts or source_ratios.")
        self.dataset = dataset
        self.global_batch_size = int(global_batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.per_rank_batch_size = int(per_rank_batch_size or self.global_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if self.global_batch_size <= 0 or self.per_rank_batch_size <= 0:
            raise ValueError("Batch sizes must be positive.")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive.")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}.")
        if self.global_batch_size != self.per_rank_batch_size * self.world_size:
            raise ValueError(
                "global_batch_size must equal per_rank_batch_size * world_size for fixed-ratio mixed batches, "
                f"got {self.global_batch_size} != {self.per_rank_batch_size} * {self.world_size}."
            )

        if source_counts is not None:
            counts = _normalize_source_counts(source_counts, dataset.names)
        else:
            counts = _counts_from_ratios(source_ratios or [], dataset.names, self.global_batch_size)
        if any(count < 0 for count in counts):
            raise ValueError(f"batch_source_counts must be non-negative, got {counts}.")
        if sum(counts) != self.global_batch_size:
            raise ValueError(
                f"batch_source_counts must sum to global batch size {self.global_batch_size}, got {counts}."
            )
        for source_id, count in enumerate(counts):
            if count > 0 and len(dataset.datasets[source_id]) <= 0:
                raise ValueError(f"Mixed source {dataset.names[source_id]!r} has no samples.")
        self.source_counts = counts
        self.local_source_counts = (
            [count // self.world_size for count in counts]
            if all(count % self.world_size == 0 for count in counts)
            else None
        )

        if num_batches is None:
            if self.drop_last:
                num_batches = len(dataset) // self.global_batch_size
            else:
                num_batches = (len(dataset) + self.global_batch_size - 1) // self.global_batch_size
        self.num_batches = max(int(num_batches), 0)

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1_000_003)
        start = self.rank * self.per_rank_batch_size
        end = start + self.per_rank_batch_size
        for _ in range(self.num_batches):
            if self.local_source_counts is not None:
                local_generator = torch.Generator()
                local_generator.manual_seed(int(torch.randint(2**31 - 1, (1,), generator=generator).item()) + self.rank)
                local_batch: list[tuple[int, int]] = []
                for source_id, count in enumerate(self.local_source_counts):
                    if count <= 0:
                        continue
                    source_len = len(self.dataset.datasets[source_id])
                    indices = torch.randint(source_len, (count,), generator=local_generator).tolist()
                    local_batch.extend((source_id, int(source_index)) for source_index in indices)
                if self.shuffle:
                    order = torch.randperm(len(local_batch), generator=local_generator).tolist()
                    local_batch = [local_batch[index] for index in order]
                yield local_batch
                continue

            global_batch: list[tuple[int, int]] = []
            for source_id, count in enumerate(self.source_counts):
                if count <= 0:
                    continue
                source_len = len(self.dataset.datasets[source_id])
                indices = torch.randint(source_len, (count,), generator=generator).tolist()
                global_batch.extend((source_id, int(source_index)) for source_index in indices)
            if self.shuffle:
                order = torch.randperm(len(global_batch), generator=generator).tolist()
                global_batch = [global_batch[index] for index in order]
            yield global_batch[start:end]


class MixedRawDataset(Dataset):
    """Weighted mixture of RAW training datasets with a stable batch schema."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Sequence[float] | None = None,
        names: Sequence[str] | None = None,
        length: int = 0,
        deterministic: bool = False,
        base_seed: int = 0,
    ) -> None:
        if not datasets:
            raise ValueError("MixedRawDataset requires at least one source dataset.")
        self.datasets = list(datasets)
        self.names = list(names or [f"source_{index}" for index in range(len(self.datasets))])
        if len(self.names) != len(self.datasets):
            raise ValueError("names length must match datasets length.")
        weight_values = [1.0] * len(self.datasets) if weights is None else [float(value) for value in weights]
        if len(weight_values) != len(self.datasets):
            raise ValueError("weights length must match datasets length.")
        if any(value < 0.0 for value in weight_values) or sum(weight_values) <= 0.0:
            raise ValueError(f"MixedRawDataset weights must be non-negative and not all zero, got {weight_values}.")
        self.weights = torch.tensor(weight_values, dtype=torch.float32)
        self.weights = self.weights / self.weights.sum()
        self.length = int(length) if int(length) > 0 else sum(len(dataset) for dataset in self.datasets)
        self.deterministic = bool(deterministic)
        self.base_seed = int(base_seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int | tuple[int, int] | list[int]) -> dict[str, torch.Tensor]:
        if isinstance(index, (tuple, list)) and len(index) == 2:
            dataset_id = int(index[0])
            source_index = int(index[1])
            if self.deterministic:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(self.base_seed + dataset_id * 1_000_003 + source_index)
                    return self._make_sample_from_source(dataset_id, source_index)
            return self._make_sample_from_source(dataset_id, source_index)
        if self.deterministic:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.base_seed + int(index))
                return self._make_sample(index)
        return self._make_sample(index)

    def _make_sample(self, index: int) -> dict[str, torch.Tensor]:
        dataset_id = int(torch.multinomial(self.weights, 1).item())
        source_index = int(torch.randint(0, len(self.datasets[dataset_id]), (1,)).item())
        return self._make_sample_from_source(dataset_id, source_index)

    def _make_sample_from_source(self, dataset_id: int, source_index: int) -> dict[str, torch.Tensor]:
        if not (0 <= dataset_id < len(self.datasets)):
            raise IndexError(f"MixedRawDataset dataset_id={dataset_id} out of range.")
        dataset = self.datasets[dataset_id]
        source_index = int(source_index) % len(dataset)
        sample = dataset[source_index]
        if not isinstance(sample, dict):
            raise TypeError(f"MixedRawDataset source {self.names[dataset_id]!r} returned {type(sample)!r}, expected dict.")
        return self._normalize_sample(sample, dataset_id=dataset_id, source_index=source_index)

    def _normalize_sample(
        self,
        sample: dict,
        *,
        dataset_id: int,
        source_index: int,
    ) -> dict[str, torch.Tensor]:
        required = ("clean_raw", "noisy_raw", "noise_map", "linear_rgb")
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"MixedRawDataset source {self.names[dataset_id]!r} missing keys: {missing}.")

        out: dict[str, torch.Tensor] = {}
        for key in ("clean_raw", "noisy_raw", "noise_map", "linear_rgb"):
            value = sample[key]
            if not torch.is_tensor(value):
                raise TypeError(f"MixedRawDataset key {key!r} must be a tensor, got {type(value)!r}.")
            out[key] = value.contiguous()

        camera_rgb = sample.get("camera_rgb", out["linear_rgb"])
        out["camera_rgb"] = camera_rgb.contiguous() if torch.is_tensor(camera_rgb) else out["linear_rgb"]
        out["noise_variance"] = out["noise_map"].square().contiguous()
        out["noise_sigma"] = out["noise_map"].contiguous()

        awb_gains = sample.get("awb_gains")
        out["awb_gains"] = (
            awb_gains.float().contiguous()
            if torch.is_tensor(awb_gains)
            else torch.ones(3, dtype=torch.float32)
        )
        iso = sample.get("iso")
        out["iso"] = iso.float().reshape(()) if torch.is_tensor(iso) else torch.tensor(0.0, dtype=torch.float32)
        analog_gain = sample.get("analog_gain")
        out["analog_gain"] = (
            analog_gain.float().reshape(())
            if torch.is_tensor(analog_gain)
            else (out["iso"] / 100.0).clamp_min(1.0)
        )
        gain_index = sample.get("gain_index")
        out["gain_index"] = (
            gain_index.long().reshape(())
            if torch.is_tensor(gain_index)
            else torch.tensor(-1, dtype=torch.long)
        )
        exposure_scale = sample.get("exposure_scale")
        out["exposure_scale"] = (
            exposure_scale.float().reshape(())
            if torch.is_tensor(exposure_scale)
            else torch.tensor(1.0, dtype=torch.float32)
        )
        image_id = sample.get("image_id")
        out["image_id"] = (
            image_id.long().reshape(())
            if torch.is_tensor(image_id)
            else torch.tensor(int(source_index), dtype=torch.long)
        )

        for key, dtype in (
            ("codec_bpp", torch.float32),
            ("codec_encoded_bytes", torch.long),
            ("codec_cache_hit", torch.bool),
            ("codec_index", torch.long),
        ):
            value = sample.get(key)
            if torch.is_tensor(value):
                out[key] = value.to(dtype=dtype).reshape(())
        uncompressed_noisy_raw = sample.get("uncompressed_noisy_raw")
        if torch.is_tensor(uncompressed_noisy_raw):
            out["uncompressed_noisy_raw"] = uncompressed_noisy_raw.contiguous()
        out["dataset_id"] = torch.tensor(int(dataset_id), dtype=torch.long)
        return out
