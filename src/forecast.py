"""Tesla next-day price forecasting, tested honestly against a naive baseline.

Part 1 reproduces the original approach (15-day price windows, min-max
scaling, tree models) and shows why it fails on a rising stock.
Part 2 predicts daily *returns* instead of prices, which any model can do
without leaving its training range, and compares every model with the
simplest possible forecast: tomorrow's price = today's price.

    python src/forecast.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "TSLA.csv"
RESULTS, IMAGES = ROOT / "results", ROOT / "images"
WINDOW = 15          # days of history per prediction, as in the original notebook
TRAIN_SHARE = 0.70   # first 70% of days for training, as in the original notebook


def load(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def split_index(n: int, share: float = TRAIN_SHARE) -> int:
    return int(n * share)


def windows(values: np.ndarray, width: int = WINDOW):
    """X[i] = values[i : i+width], y[i] = values[i+width] (the next day)."""
    X = np.array([values[i:i + width] for i in range(len(values) - width)])
    y = values[width:]
    return X, y


def usd_errors(actual, predicted) -> dict:
    return {"mae_usd": round(float(mean_absolute_error(actual, predicted)), 2),
            "rmse_usd": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2)}


# ---------- Part 1: the original approach ----------

def original_approach(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    close = df["close"].to_numpy()
    cut = split_index(len(close))
    # Fit the scaler on training days only (the original fitted on all days, which leaks the
    # test period's highs into training; it doesn't cause the failure below, but it's still wrong).
    scaler = MinMaxScaler().fit(close[:cut].reshape(-1, 1))
    scaled = scaler.transform(close.reshape(-1, 1)).ravel()
    X_tr, y_tr = windows(scaled[:cut])
    X_te, y_te = windows(scaled[cut:])
    test_dates = df["date"].iloc[cut + WINDOW:].reset_index(drop=True)
    actual = scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()

    models = {"Random forest (price windows)": RandomForestRegressor(300, random_state=0, n_jobs=-1)}
    if XGBRegressor is not None:
        models = {"XGBoost (price windows, original)": XGBRegressor(n_estimators=1000, random_state=0),
                  **models}
    rows, preds = [], {"date": test_dates, "actual": actual, "naive": close[cut + WINDOW - 1:-1]}
    rows.append({"part": "1: price windows", "model": "Naive: tomorrow = today", **usd_errors(actual, preds["naive"])})
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        p = scaler.inverse_transform(model.predict(X_te).reshape(-1, 1)).ravel()
        preds[name] = p
        rows.append({"part": "1: price windows", "model": name, **usd_errors(actual, p),
                     "highest_forecast_usd": round(float(p.max()), 2)})
    info = {"train_max_usd": float(close[:cut].max()), "test_max_usd": float(close[cut:].max()),
            "split_date": str(df["date"].iloc[cut].date())}
    return pd.DataFrame(preds), {"rows": rows, **info}


# ---------- Part 2: predict returns instead ----------

def returns_table(df: pd.DataFrame) -> pd.DataFrame:
    """Row t holds the last WINDOW daily log returns up to day t, and the return from t to t+1."""
    out = pd.DataFrame({"date": df["date"], "close": df["close"]})
    out["ret"] = np.log(out["close"]).diff()
    for k in range(WINDOW):
        out[f"ret_lag{k}"] = out["ret"].shift(k)          # lag0 = today's return
    out["next_ret"] = out["ret"].shift(-1)
    out["next_close"] = out["close"].shift(-1)
    return out.dropna().reset_index(drop=True)


FEATURES = [f"ret_lag{k}" for k in range(WINDOW)]


def returns_approach(df: pd.DataFrame) -> list:
    table = returns_table(df)
    # Test on exactly the same days as Part 1, so the two parts are directly comparable.
    test_start = df["date"].iloc[split_index(len(df)) + WINDOW - 1]
    tr, te = table[table["date"] < test_start], table[table["date"] >= test_start]
    actual = te["next_close"].to_numpy()
    up = (te["next_ret"] > 0).to_numpy()

    rows = [{"part": "2: returns", "model": "Naive: tomorrow = today",
             **usd_errors(actual, te["close"]), "direction_accuracy": None},
            {"part": "2: returns", "model": "Always predict 'up'",
             "direction_accuracy": round(float(up.mean()), 3)}]
    models = {"Ridge regression": Ridge(alpha=1.0),
              "Random forest": RandomForestRegressor(300, min_samples_leaf=5, random_state=0, n_jobs=-1)}
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    for name, model in models.items():
        model.fit(tr[FEATURES], tr["next_ret"])
        pred_ret = model.predict(te[FEATURES])
        pred_close = te["close"].to_numpy() * np.exp(pred_ret)
        rows.append({"part": "2: returns", "model": name, **usd_errors(actual, pred_close),
                     "direction_accuracy": round(float(((pred_ret > 0) == up).mean()), 3)})
    return rows


# ---------- Charts ----------

def plot_original(preds: pd.DataFrame, info: dict):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(preds["date"], preds["actual"], color="#1f2a37", linewidth=1.6, label="Actual close")
    model_cols = [c for c in preds.columns if c not in ("date", "actual", "naive")]
    for col, color in zip(model_cols, ["#b4473c", "#e0a43b"]):
        ax.plot(preds["date"], preds[col], color=color, linewidth=1.3, label=col)
    ax.axhline(info["train_max_usd"], color="#9aa6b5", linestyle="--", linewidth=1)
    ax.text(preds["date"].iloc[5], info["train_max_usd"] + 18,
            f"highest price in training data: ${info['train_max_usd']:.0f}", color="#5b6675", fontsize=9)
    ax.set_ylabel("USD")
    ax.set_title("Tree models can't forecast above the prices they were trained on", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / "original_approach.png", dpi=150)
    plt.close(fig)


def plot_errors(results: pd.DataFrame, period: str):
    r = results.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#9aa6b5" if m.startswith(("Naive", "Always")) else
              ("#b4473c" if "price windows" in m else "#1f5f8b") for m in r["model"]]
    labels = [f"{m}  [{p.split(':')[0]}]" for m, p in zip(r["model"], r["part"])]
    ax.barh(labels, r["mae_usd"], color=colors)
    for y, v in enumerate(r["mae_usd"]):
        ax.text(v + 4, y, f"${v:,.2f}", va="center", fontsize=9)
    ax.set_xlim(0, r["mae_usd"].max() * 1.15)
    ax.set_xlabel("Average error per day, USD")
    ax.set_title(f"Next-day forecast error, {period}", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / "forecast_errors.png", dpi=150)
    plt.close(fig)


def main() -> pd.DataFrame:
    df = load()
    preds, info = original_approach(df)
    rows = info.pop("rows") + returns_approach(df)
    results = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "results.csv", index=False)
    # The naive forecast is identical in both parts, so chart it once.
    chart = results.dropna(subset=["mae_usd"]).drop_duplicates(subset=["model", "mae_usd"])
    period = f"{preds['date'].iloc[0]:%b %Y} to {preds['date'].iloc[-1]:%b %Y}"
    plot_errors(chart, period)
    plot_original(preds, info)
    print(f"Split on {info['split_date']}: training prices up to ${info['train_max_usd']:.2f}, "
          f"test prices up to ${info['test_max_usd']:.2f}")
    with pd.option_context("display.width", 200):
        print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
