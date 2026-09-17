from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_image

from deeptrace_ml.audit.features import PIXEL_FEATURES, metadata_feature_frame, pixel_feature_frame, pixel_features
from deeptrace_ml.audit.spectrum import centered_log_power, mean_spectrum, radial_profile
from deeptrace_ml.audit.trivial_classifier import run_trivial_classifier


def test_trivial_classifier_detects_planted_shortcut():
    rng = np.random.default_rng(0)
    n = 400
    labels = pd.Series(["real"] * n + ["fake"] * n)
    X = pd.DataFrame(
        {
            "width": np.where(labels == "fake", 1024, 256),  # perfect shortcut
            "noise": rng.normal(size=2 * n),
            "constant": 1.0,
        }
    )
    res = run_trivial_classifier(X, labels, "planted", seed=0)
    assert res.auc["gradient_boosting"]["mean"] > 0.99
    assert res.top_features[0]["feature"] == "width"
    assert res.constant_features_dropped == ["constant"]


def test_trivial_classifier_near_chance_on_noise():
    rng = np.random.default_rng(1)
    n = 400
    labels = pd.Series(["real"] * n + ["fake"] * n)
    X = pd.DataFrame(rng.normal(size=(2 * n, 3)), columns=["a", "b", "c"])
    res = run_trivial_classifier(X, labels, "noise", seed=0)
    assert abs(res.auc["logistic_regression"]["mean"] - 0.5) < 0.1


def test_trivial_classifier_all_constant():
    labels = pd.Series(["real"] * 10 + ["fake"] * 10)
    res = run_trivial_classifier(pd.DataFrame({"w": [224] * 20}), labels, "const", seed=0)
    assert res.n_features_used == 0 and res.auc["logistic_regression"]["mean"] == 0.5


def test_trivial_classifier_needs_both_classes():
    with pytest.raises(ValueError):
        run_trivial_classifier(pd.DataFrame({"a": range(10)}), pd.Series(["real"] * 10), "x", seed=0)


def test_metadata_feature_frame():
    manifest = pd.DataFrame(
        {
            "width": [256, 1024],
            "height": [256, 1024],
            "file_size": [20_000, 1_500_000],
            "jpeg_quality_est": [95, None],
            "format": ["jpeg", "png"],
            "has_exif": [False, "True"],
            "has_icc": [False, False],
            "has_xmp": [True, False],
        }
    )
    feats = metadata_feature_frame(manifest)
    assert feats["is_png"].tolist() == [0, 1]
    assert feats["jpeg_quality_est"].tolist() == [95, -1]
    assert feats["jpeg_quality_missing"].tolist() == [0, 1]
    assert feats["has_exif"].tolist() == [0, 1]


def test_pixel_features_and_frame(tmp_path):
    feats = pixel_features(make_image(64, 64))
    assert set(feats) == set(PIXEL_FEATURES)
    assert 0 <= feats["hf_energy_ratio"] <= 1
    make_image(64, 64).save(tmp_path / "ok.png")
    (tmp_path / "bad.png").write_bytes(b"x")
    frame = pixel_feature_frame([tmp_path / "ok.png", tmp_path / "bad.png"], workers=1)
    assert frame.iloc[0].notna().all() and frame.iloc[1].isna().all()


def test_spectrum_shapes_and_small_images(tmp_path):
    assert centered_log_power(make_image(100, 100), 224) is None
    spec = centered_log_power(make_image(256, 256), 224)
    assert spec.shape == (224, 224)
    assert radial_profile(spec).shape == (112,)
    make_image(256, 256, seed=1).save(tmp_path / "big.png")
    make_image(64, 64).save(tmp_path / "small.png")
    result = mean_spectrum([tmp_path / "big.png", tmp_path / "small.png"], size=224, workers=1)
    assert (result.n_used, result.n_skipped) == (1, 1)


def test_radial_profile_detects_periodic_pattern():
    # A pure horizontal cosine puts energy at one radius: the profile must peak there.
    size, freq = 128, 20
    x = np.cos(2 * np.pi * freq * np.arange(size) / size)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(np.tile(x, (size, 1))))) ** 2
    profile = radial_profile(np.log(spectrum + 1e-8))
    assert int(np.argmax(profile)) == freq
