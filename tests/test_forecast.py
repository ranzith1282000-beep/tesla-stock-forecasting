"""Checks that the forecasts never use information from the future."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import forecast as f  # noqa: E402


def test_data():
    df = f.load()
    assert len(df) == 1258 and df["date"].is_monotonic_increasing
    assert df.isna().sum().sum() == 0


def test_windows_target_is_next_value():
    X, y = f.windows(np.arange(20.0), width=5)
    assert X[0].tolist() == [0, 1, 2, 3, 4] and y[0] == 5
    assert len(X) == len(y) == 15


def test_returns_features_only_use_the_past():
    df = f.load()
    base = f.returns_table(df)
    altered = df.copy()
    cut = altered["date"] >= "2020-06-01"
    altered.loc[cut, "close"] *= 2
    changed = f.returns_table(altered)
    early = base["date"] < "2020-05-29"  # a couple of days before the change
    assert np.allclose(base.loc[early, f.FEATURES], changed.loc[early, f.FEATURES])


def test_next_close_is_tomorrow():
    t = f.returns_table(f.load())
    assert (t["next_close"].iloc[:-1].to_numpy() == t["close"].iloc[1:].to_numpy()).all()


def test_train_and_test_do_not_overlap():
    df = f.load()
    preds, info = f.original_approach(df)
    assert preds["date"].min() > df["date"].iloc[f.split_index(len(df)) - 1]
