#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

FILE28 = "pnl/factor28-combo"
FILE36 = "pnl/factor36-combo"
OUTDIR = "pnl_book_combo"

TRADING_DAYS = 252
RISK_WINDOW = 252
RISK_MIN_PERIODS = 60


def load_pnl(path, prefix):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] < 5:
        raise ValueError("%s expected >=5 columns" % path)

    return pd.DataFrame({
        "date": df.iloc[:, 0].astype(np.int64),
        prefix + "_pnl": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
        prefix + "_ret": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
    })


def max_drawdown(ret):
    s = pd.Series(ret, dtype=float).dropna().values
    if len(s) == 0:
        return np.nan
    curve = np.cumsum(s)
    peak = np.maximum.accumulate(np.r_[0.0, curve])[1:]
    dd = curve - peak
    return -float(dd.min())


def perf_stats(ret):
    s = pd.Series(ret, dtype=float).dropna()
    mean = s.mean()
    std = s.std(ddof=1)

    return {
        "n": len(s),
        "ann_ret_pct": mean * TRADING_DAYS * 100.0,
        "sharpe": (
            mean / std * np.sqrt(TRADING_DAYS)
            if np.isfinite(std) and std > 0
            else np.nan
        ),
        "max_dd_pct": max_drawdown(s.values) * 100.0,
        "win_rate": float((s > 0).mean()),
    }


def summarize(df, ret_col, name):
    x = df.copy()
    x["year"] = x["date"].astype(str).str[:4].astype(int)
    rows = []

    for year, g in x.groupby("year"):
        rows.append({
            "factor": name,
            "period": str(year),
            **perf_stats(g[ret_col]),
        })

    rows.append({
        "factor": name,
        "period": "%d-%d" % (x["date"].min(), x["date"].max()),
        **perf_stats(x[ret_col]),
    })

    return rows


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    a = load_pnl(FILE28, "f28")
    b = load_pnl(FILE36, "f36")

    df = (
        a.merge(b, on="date", how="inner", validate="one_to_one")
         .sort_values("date")
         .reset_index(drop=True)
    )

    if len(df) == 0:
        raise RuntimeError("No overlapping dates")

    corr = df[["f28_ret", "f36_ret"]].corr().iloc[0, 1]

    combos = {
        "factor41-book-85_15": (0.85, 0.15),
        "factor42-book-75_25": (0.75, 0.25),
        "factor43-book-65_35": (0.65, 0.35),
    }

    for name, (w28, w36) in combos.items():
        df[name + "_ret"] = w28 * df["f28_ret"] + w36 * df["f36_ret"]
        df[name + "_w28"] = w28
        df[name + "_w36"] = w36

    # Factor44: ex-ante inverse-vol weights.
    # shift(1) prevents look-ahead.
    vol28 = df["f28_ret"].rolling(
        RISK_WINDOW, min_periods=RISK_MIN_PERIODS
    ).std().shift(1)

    vol36 = df["f36_ret"].rolling(
        RISK_WINDOW, min_periods=RISK_MIN_PERIODS
    ).std().shift(1)

    inv28 = 1.0 / vol28.replace(0.0, np.nan)
    inv36 = 1.0 / vol36.replace(0.0, np.nan)
    denom = inv28 + inv36

    w28 = (inv28 / denom).fillna(0.5)
    w36 = (inv36 / denom).fillna(0.5)

    name44 = "factor44-book-risk-parity"
    df[name44 + "_ret"] = w28 * df["f28_ret"] + w36 * df["f36_ret"]
    df[name44 + "_w28"] = w28
    df[name44 + "_w36"] = w36

    names = list(combos.keys()) + [name44]

    for name in names:
        pd.DataFrame({
            "date": df["date"],
            "ret": df[name + "_ret"],
            "cum_ret": df[name + "_ret"].cumsum(),
            "w28": df[name + "_w28"],
            "w36": df[name + "_w36"],
        }).to_csv(
            os.path.join(OUTDIR, name + ".csv"),
            index=False
        )

    summary_rows = []

    for name, col in [
        ("factor28-book", "f28_ret"),
        ("factor36-book", "f36_ret"),
    ]:
        summary_rows += summarize(df, col, name)

    for name in names:
        summary_rows += summarize(df, name + "_ret", name)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        os.path.join(OUTDIR, "book_combo_summary.csv"),
        index=False
    )

    full = summary.groupby("factor", sort=False).tail(1)

    print("=" * 78)
    print("INDEPENDENT BOOK COMBINATION")
    print("=" * 78)
    print("dates:", int(df["date"].iloc[0]), "->", int(df["date"].iloc[-1]))
    print("rows:", len(df))
    print("corr(F28,F36): %.6f" % corr)
    print("")
    print(full[
        ["factor", "ann_ret_pct", "sharpe", "max_dd_pct", "win_rate"]
    ].to_string(
        index=False,
        formatters={
            "ann_ret_pct": "{:.3f}".format,
            "sharpe": "{:.3f}".format,
            "max_dd_pct": "{:.3f}".format,
            "win_rate": "{:.3f}".format,
        }
    ))
    print("")
    print("Factor44 avg w28: %.4f" % df[name44 + "_w28"].mean())
    print("Factor44 avg w36: %.4f" % df[name44 + "_w36"].mean())
    print("summary:", os.path.join(OUTDIR, "book_combo_summary.csv"))
    print("=" * 78)


if __name__ == "__main__":
    main()
