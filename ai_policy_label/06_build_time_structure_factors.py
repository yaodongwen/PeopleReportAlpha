#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_build_time_structure_factors.py

Build two NEW PeopleDaily time-structure factors without re-running AI.

Input
-----
ai_policy_label/outputs/industry_signed_daily.parquet

Uses the existing full-calendar grid from:
data/factor/people_daily_ai_policy_factor.npz

Output
------
data/factor/people_daily_time_structure_factor.npz

Factor51: policy_persistence_surprise_20_rank
    active_day_ratio_20
      = fraction of the last 20 NATURAL calendar days for which
        quality_attention > 0 for this SW1 industry.

    Then:
      own-industry historical z-score versus shifted 365-day history
      min_history=120
      daily cross-sectional rank -> [-1, 1] style scale.

Factor52: policy_burst_surprise_5_20_rank
    burst_5_20
      = 5-natural-day mean(quality_attention)
        - 20-natural-day mean(quality_attention)

    Then the same shifted-365 historical z-score and daily CS rank.

Important timing
----------------
The NPZ stores signal values on NATURAL calendar date t.
The Alpha runtime continues to trade on date T using T-1 calendar date,
so there is no additional shift here.
"""

import os
import numpy as np
import pandas as pd


SIGNED_DAILY_CANDIDATES = [
    "ai_policy_label/outputs/industry_signed_daily.parquet",
    "outputs/industry_signed_daily.parquet",
]

BASE_FACTOR_CANDIDATES = [
    "data/factor/people_daily_ai_policy_factor.npz",
    "../data/factor/people_daily_ai_policy_factor.npz",
]

OUTPUT_CANDIDATES = [
    "data/factor/people_daily_time_structure_factor.npz",
    "../data/factor/people_daily_time_structure_factor.npz",
]

DEBUG_CANDIDATES = [
    "ai_policy_label/outputs/time_structure_daily_debug.parquet",
    "outputs/time_structure_daily_debug.parquet",
]

PERSIST_WINDOW = 20
BURST_SHORT_WINDOW = 5
BURST_LONG_WINDOW = 20
HIST_WINDOW = 365
MIN_HISTORY = 120
STD_FLOOR = 1e-6


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def choose_output(paths):
    for p in paths:
        parent = os.path.dirname(p)
        if parent and os.path.isdir(parent):
            return p
    return paths[0]


def normalize_date(s):
    if np.issubdtype(s.dtype, np.datetime64):
        return s.dt.strftime("%Y%m%d").astype(np.int64)

    # Fast path for integer-like YYYYMMDD.
    numeric = pd.to_numeric(s, errors="coerce")
    good_numeric = numeric.notna()
    out = pd.Series(np.nan, index=s.index, dtype=np.float64)

    if good_numeric.any():
        vals = numeric.loc[good_numeric].astype(np.int64)
        looks_like_yyyymmdd = (
            (vals >= 19000101)
            & (vals <= 21001231)
        )
        if looks_like_yyyymmdd.all():
            out.loc[good_numeric] = vals.values
            return out

    parsed = pd.to_datetime(s, errors="coerce")
    ok = parsed.notna()
    out.loc[ok] = (
        parsed.loc[ok]
        .dt.strftime("%Y%m%d")
        .astype(np.int64)
    )
    return out


def cs_rank_matrix(matrix):
    """
    Project convention:
        rank(pct=True) * 2 - 1

    Shape:
        [date, industry]
    """
    ranked = (
        pd.DataFrame(matrix)
        .rank(
            axis=1,
            pct=True,
            method="average",
            na_option="keep",
        )
        * 2.0
        - 1.0
    )
    return ranked.values.astype(np.float64)


def historical_surprise_rank(raw_signal):
    """
    raw_signal shape = [natural calendar date, SW1]

    For each industry:
      z_t = (x_t - mean(x_{t-365:t-1}))
            / std(x_{t-365:t-1})

    Then daily cross-sectional rank.
    """
    n_dates, n_codes = raw_signal.shape

    surprise = np.full(
        (n_dates, n_codes),
        np.nan,
        dtype=np.float64,
    )

    for j in range(n_codes):
        s = pd.Series(raw_signal[:, j], dtype=np.float64)
        hist = s.shift(1)

        mean = hist.rolling(
            HIST_WINDOW,
            min_periods=MIN_HISTORY,
        ).mean()

        std = hist.rolling(
            HIST_WINDOW,
            min_periods=MIN_HISTORY,
        ).std()

        std = std.where(std > STD_FLOOR, np.nan)

        surprise[:, j] = ((s - mean) / std).values

    return (
        surprise,
        cs_rank_matrix(surprise),
    )


def main():
    signed_path = first_existing(SIGNED_DAILY_CANDIDATES)
    base_path = first_existing(BASE_FACTOR_CANDIDATES)

    if signed_path is None:
        raise FileNotFoundError(
            "industry_signed_daily.parquet not found. Tried:\n  "
            + "\n  ".join(SIGNED_DAILY_CANDIDATES)
        )

    if base_path is None:
        raise FileNotFoundError(
            "people_daily_ai_policy_factor.npz not found. Tried:\n  "
            + "\n  ".join(BASE_FACTOR_CANDIDATES)
        )

    output_path = choose_output(OUTPUT_CANDIDATES)
    debug_path = choose_output(DEBUG_CANDIDATES)

    print("=" * 78)
    print("BUILD PEOPLEDaily TIME-STRUCTURE FACTORS")
    print("=" * 78)
    print("signed daily:", signed_path)
    print("base npz    :", base_path)
    print("output npz  :", output_path)

    base = np.load(base_path)

    dates = base["dates"].astype(np.int64)
    codes = base["codes"].astype(np.int64)

    print(
        "grid:",
        len(dates),
        "dates x",
        len(codes),
        "industries =",
        len(dates) * len(codes),
    )
    print("date range:", int(dates[0]), "->", int(dates[-1]))

    df = pd.read_parquet(signed_path)

    required = {
        "date",
        "industry",
        "quality_attention",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(
            "signed daily missing columns %s; actual=%s"
            % (missing, list(df.columns))
        )

    df = df[[
        "date",
        "industry",
        "quality_attention",
    ]].copy()

    df["date_int"] = normalize_date(df["date"])
    df["code"] = pd.to_numeric(
        df["industry"],
        errors="coerce",
    )
    df["quality_attention"] = pd.to_numeric(
        df["quality_attention"],
        errors="coerce",
    ).fillna(0.0)

    df = df[
        df["date_int"].notna()
        & df["code"].notna()
    ].copy()

    df["date_int"] = df["date_int"].astype(np.int64)
    df["code"] = df["code"].astype(np.int64)

    valid_dates = set(dates.tolist())
    valid_codes = set(codes.tolist())

    df = df[
        df["date_int"].isin(valid_dates)
        & df["code"].isin(valid_codes)
    ].copy()

    # Defensive aggregation in case duplicates exist.
    daily = (
        df.groupby(
            ["date_int", "code"],
            as_index=False,
        )["quality_attention"]
        .sum()
    )

    full_index = pd.MultiIndex.from_product(
        [dates, codes],
        names=["date_int", "code"],
    )

    grid = (
        daily.set_index(["date_int", "code"])
        .reindex(full_index)
        .fillna(0.0)
        .reset_index()
    )

    attention = (
        grid["quality_attention"]
        .values
        .reshape(len(dates), len(codes))
        .astype(np.float64)
    )

    att_df = pd.DataFrame(attention)

    # ---------------------------------------------------------
    # Factor51: persistence
    # ---------------------------------------------------------
    active = (attention > 0.0).astype(np.float64)

    persistence_20 = (
        pd.DataFrame(active)
        .rolling(
            window=PERSIST_WINDOW,
            min_periods=1,
        )
        .mean()
        .values
        .astype(np.float64)
    )

    persistence_surprise, persistence_rank = (
        historical_surprise_rank(persistence_20)
    )

    # ---------------------------------------------------------
    # Factor52: burst
    # ---------------------------------------------------------
    mean5 = (
        att_df.rolling(
            window=BURST_SHORT_WINDOW,
            min_periods=1,
        )
        .mean()
        .values
    )

    mean20 = (
        att_df.rolling(
            window=BURST_LONG_WINDOW,
            min_periods=1,
        )
        .mean()
        .values
    )

    burst_5_20 = (
        mean5 - mean20
    ).astype(np.float64)

    burst_surprise, burst_rank = (
        historical_surprise_rank(burst_5_20)
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        dates=dates,
        codes=codes,
        policy_persistence_surprise_20_rank=persistence_rank,
        policy_burst_surprise_5_20_rank=burst_rank,
        # Raw/debug matrices are useful for later research and cheap to keep.
        active_day_ratio_20=persistence_20,
        policy_burst_5_20=burst_5_20,
        policy_persistence_surprise_20=persistence_surprise,
        policy_burst_surprise_5_20=burst_surprise,
    )

    # Debug long-form parquet.
    try:
        debug = pd.DataFrame({
            "date": np.repeat(dates, len(codes)),
            "industry": np.tile(codes, len(dates)),
            "quality_attention": attention.reshape(-1),
            "active_day_ratio_20": persistence_20.reshape(-1),
            "persistence_surprise": persistence_surprise.reshape(-1),
            "persistence_rank": persistence_rank.reshape(-1),
            "burst_5_20": burst_5_20.reshape(-1),
            "burst_surprise": burst_surprise.reshape(-1),
            "burst_rank": burst_rank.reshape(-1),
        })

        os.makedirs(
            os.path.dirname(debug_path),
            exist_ok=True,
        )

        debug.to_parquet(
            debug_path,
            index=False,
        )

        print("debug parquet:", debug_path)

    except Exception as e:
        print("WARNING: debug parquet not written:", repr(e))

    def finite_ratio(x):
        return float(np.isfinite(x).mean())

    print("")
    print("finite ratios:")
    print(
        "  persistence rank:",
        "%.4f" % finite_ratio(persistence_rank)
    )
    print(
        "  burst rank      :",
        "%.4f" % finite_ratio(burst_rank)
    )

    # First date with any finite values.
    for name, mat in [
        ("persistence", persistence_rank),
        ("burst", burst_rank),
    ]:
        ok = np.isfinite(mat).any(axis=1)
        idx = np.where(ok)[0]
        first = int(dates[idx[0]]) if len(idx) else None
        print("  first finite %-11s:" % name, first)

    print("")
    print("saved:", output_path)
    print("fields:")
    with np.load(output_path) as z:
        for k in z.files:
            print(" ", k, z[k].shape, z[k].dtype)
    print("=" * 78)


if __name__ == "__main__":
    main()
