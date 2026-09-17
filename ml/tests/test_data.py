from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_image

from deeptrace_ml.data.datasets import find_140k_root, specs_140k, specs_external
from deeptrace_ml.data.dedupe import (
    compute_hashes,
    find_near_duplicates,
    popcount64,
    resolve_drops,
)
from deeptrace_ml.data.manifest import build_manifest, load_manifest, parse_roots, save_manifest
from deeptrace_ml.data.splits import split_validation

# ---------- dedupe ----------


def test_popcount64():
    values = np.array([0, 1, 0xFF, 2**64 - 1], dtype=np.uint64)
    assert popcount64(values).tolist() == [0, 1, 8, 64]


def _brute_force(hashes: np.ndarray, d: int) -> set[tuple[int, int]]:
    pairs = set()
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dp = bin(int(hashes[i, 0]) ^ int(hashes[j, 0])).count("1")
            dd = bin(int(hashes[i, 1]) ^ int(hashes[j, 1])).count("1")
            if dp <= d and dd <= d:
                pairs.add((i, j))
    return pairs


@pytest.mark.parametrize("max_distance", [0, 2, 4])
def test_banded_search_matches_brute_force(max_distance):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**63, size=(60, 2), dtype=np.int64).astype(np.uint64)
    # plant near-duplicates by flipping a few bits
    near = base[:20].copy()
    for row in near:
        for col in range(2):
            for bit in rng.choice(64, size=rng.integers(0, 6), replace=False):
                row[col] ^= np.uint64(1) << np.uint64(bit)
    hashes = np.vstack([base, near])
    found = find_near_duplicates(hashes, max_distance)
    assert set(zip(found["i"], found["j"], strict=False)) == _brute_force(hashes, max_distance)


def test_resolve_drops_prefers_dropping_test_side():
    manifest = pd.DataFrame(
        {
            "dataset": ["140k", "140k", "140k", "crossgen"],
            "split": ["train", "test", "train", "crossgen"],
            "rel_path": ["a.jpg", "b.jpg", "c.jpg", "d.png"],
            "label": ["real", "real", "fake", "real"],
        }
    )
    pairs = pd.DataFrame({"i": [0, 0, 1], "j": [1, 2, 3], "phash_dist": [1, 0, 2], "dhash_dist": [1, 0, 2]})
    annotated, drops = resolve_drops(manifest, pairs, ["train", "val_select", "val_calib", "test", "crossgen"])
    assert annotated["cross_split"].tolist() == [True, False, True]
    assert annotated["label_conflict"].tolist() == [False, True, False]
    assert set(drops["rel_path"]) == {"b.jpg", "d.png"}


def test_resolve_drops_unknown_split_raises():
    manifest = pd.DataFrame({"dataset": ["x"], "split": ["mystery"], "rel_path": ["a"], "label": ["real"]})
    with pytest.raises(ValueError, match="mystery"):
        resolve_drops(manifest, pd.DataFrame(columns=["i", "j", "phash_dist", "dhash_dist"]), ["train"])


def test_compute_hashes_identical_images_match(tmp_path):
    img = make_image(128, 128, seed=3)
    img.save(tmp_path / "a.png")
    img.save(tmp_path / "b.jpg", quality=90)  # re-encoded copy should still be a near-duplicate
    make_image(128, 128, seed=99).save(tmp_path / "c.png")
    (tmp_path / "broken.png").write_bytes(b"not an image")
    hashes, ok = compute_hashes([tmp_path / n for n in ("a.png", "b.jpg", "c.png", "broken.png")], workers=1)
    assert ok.tolist() == [True, True, True, False]
    pairs = find_near_duplicates(hashes[ok], max_distance=4)
    assert (0, 1) in set(zip(pairs["i"], pairs["j"], strict=False))


# ---------- splits ----------


def _valid_frame(n_per_label: int) -> pd.DataFrame:
    rows = [
        {"label": label, "split": "valid", "sha256": f"{label}{i:060d}", "rel_path": f"valid/{label}/{i}.jpg"}
        for label in ("real", "fake")
        for i in range(n_per_label)
    ]
    rows.append({"label": "real", "split": "train", "sha256": "t" * 64, "rel_path": "train/real/0.jpg"})
    return pd.DataFrame(rows)


def test_split_validation_is_balanced_deterministic_and_order_independent():
    df = _valid_frame(101)
    a = split_validation(df, 42, "valid", "val_select", "val_calib")
    b = split_validation(df.sample(frac=1, random_state=1), 42, "valid", "val_select", "val_calib").loc[a.index]
    assert (a["split"] == b["split"]).all()
    counts = a.groupby(["label", "split"]).size()
    assert counts[("real", "val_select")] == 50 and counts[("real", "val_calib")] == 51
    assert a.loc[a["rel_path"] == "train/real/0.jpg", "split"].item() == "train"
    c = split_validation(df, 7, "valid", "val_select", "val_calib")
    assert not (a["split"] == c["split"]).all()  # seed matters


# ---------- datasets + manifest ----------


def test_find_140k_and_build_manifest(tmp_path):
    root = tmp_path / "input" / "140k" / "real_vs_fake" / "real-vs-fake"
    seed = 0
    for split in ("train", "valid", "test"):
        for label in ("real", "fake"):
            folder = root / split / label
            folder.mkdir(parents=True)
            for i in range(3):
                seed += 1
                make_image(32, 32, seed=seed).save(folder / f"{i}.jpg", quality=90)
    (root / "train" / "fake" / "3.jpg").write_bytes(b"corrupt")

    found = find_140k_root(tmp_path / "input")
    assert found == root
    specs = specs_140k(found, found, "140k", "ffhq", "stylegan")
    assert len(specs) == 19
    df = build_manifest(found, specs, workers=1)
    assert df["read_error"].notna().sum() == 1
    assert set(df["source"]) == {"ffhq", "stylegan"}

    path = save_manifest(df, tmp_path / "m.csv.gz")
    loaded = load_manifest(path)
    assert len(loaded) == 19 and loaded["sha256"].str.len().dropna().eq(64).all()


def test_find_140k_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_140k_root(tmp_path)


def test_specs_external_layout(tmp_path):
    for label, source in (("real", "celebahq"), ("fake", "sdxl")):
        folder = tmp_path / label / source
        folder.mkdir(parents=True)
        make_image(16, 16).save(folder / "x.png")
    specs = specs_external(tmp_path, "crossgen")
    expected = {("real", "celebahq", "crossgen"), ("fake", "sdxl", "crossgen")}
    assert {(s.label, s.source, s.split) for s in specs} == expected


def test_parse_roots_windows_path():
    roots = parse_roots([r"own=D:\photos\set"])
    assert str(roots["own"]).endswith("set")
    with pytest.raises(ValueError):
        parse_roots(["no-equals-sign"])
