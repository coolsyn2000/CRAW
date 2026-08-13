from __future__ import annotations

from pathlib import Path

from raw_codec import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_paper_config_merges_and_expands(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    config = load_config(ROOT / "configs/experiments/ours_m_lambda6e4_100k.yaml")

    assert config["model"]["name"] == "ours_m"
    assert config["loss"]["lambda_rate"] == 6.0e-4
    assert config["data"]["train"]["batch_source_counts"] == [8, 24]
    assert config["data"]["validation_sets"]["lsdir_high_texture"]["length"] == 1024
    assert config["train"]["checkpoint_dir"].startswith(str(ROOT))

