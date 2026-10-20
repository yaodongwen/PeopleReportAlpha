#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09_build_direction_consistency_factor.py

Factor72: policy direction consistency

Core idea
---------
For each SW1 industry and natural calendar day, use the signed policy
attention signal already produced by industry_signed_daily.parquet.

Over the most recent 20 natural days:

    consistency_20
      = abs(sum(signed_attention))
        / sum(abs(signed_attention))

Interpretation:
    ~1.0  -> policy direction is repeatedly confirmed in one direction
    ~0.0  -> positive/negative policy signals offset each other

This is deliberately different from:
    - burst: short-term intensity acceleration
    - concentration: whether attention is concentrated on a few dates
    - novelty: whether semantics are new

Then:
    own-industry shifted 365-natural-day historical mean/std
    min_history = 120
    -> surprise z-score
    -> daily cross-sectional rank in [-1, 1]

Timing
------
NPZ value is stored on natural day t.
Runtime trades on T using natural calendar T-1, same as existing factors.
"""

import os
import numpy as np
import pandas as pd


SIGNED_CANDIDATES = [
    "ai_policy_label/outputs/industry_signed_daily.parquet",
    "outputs/industry_signed_daily.parquet",
]

BASE_CANDIDATES = [
    "data/factor/people_daily_ai_policy_factor.npz",
    "../data/factor/people_daily_ai_policy_factor.npz",
]

OUTPUT_CANDIDATES = [
    "data/factor/people_daily_direction_consistency_factor.npz",
    "../data/factor/people_daily_direction_consistency_factor.npz",
]

DEBUG_CANDIDATES = [
    "ai_policy_label/outputs/direction_consistency_daily_debug.parquet",
    "outputs/direction_consistency_daily_debug.parquet",
]

ROLL_WINDOW = 20
HIST_WINDOW = 365
MIN_HISTORY = 120
STD_FLOOR = 1e-6
EPS = 1e-12


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def choose_output(paths):
    for p in paths:
        parent = os.path.dirname(p)
        if not parent or os.path.isdir(parent):
            return p
    return paths[0]


def normalize_date(s):
    numeric = pd.to_numeric(s, errors="coerce")
    good = numeric.notna()

    if good.any():
        vals = numeric.loc[good].astype(np.int64)
        if ((vals >= 19000101) & (vals <= 21001231)).all():
            out = pd.Series(np.nan, index=s.index, dtype=float)
            out.loc[good] = vals.values
            return out

    p = pd.to_datetime(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    ok = p.notna()
    out.loc[ok] = p.loc[ok].dt.strftime("%Y%m%d").astype(np.int64)
    return out


def detect_signed_attention_column(columns):
    """
    Prefer the already-normalized signed attention field.
    Fall back only to closely related signed fields.
    """
    candidates = [
        "signed_attention",
        "net_attention",
        "signed_quality_attention",
        "signed",
        "signed_weight",
        "net_weight",
    ]

    for c in candidates:
        if c in columns:
            return c

    raise KeyError(
        "No signed policy column found. "
        "Expected one of %s; actual columns=%s"
        % (candidates, list(columns))
    )


def cs_rank_matrix(matrix):
    return (
        pd.DataFrame(matrix)
        .rank(
            axis=1,
            pct=True,
            method="average",
            na_option="keep",
        )
        .mul(2.0)
        .sub(1.0)
        .values
        .astype(np.float64)
    )


def historical_surprise_rank(raw):
    n_dates, n_codes = raw.shape

    surprise = np.full(
        (n_dates, n_codes),
        np.nan,
        dtype=np.float64,
    )

    for j in range(n_codes):
        s = pd.Series(raw[:, j], dtype=np.float64)
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

    return surprise, cs_rank_matrix(surprise)


def main():
    signed_path = first_existing(SIGNED_CANDIDATES)
    base_path = first_existing(BASE_CANDIDATES)

    if signed_path is None:
        raise FileNotFoundError("industry_signed_daily.parquet not found")
    if base_path is None:
        raise FileNotFoundError("people_daily_ai_policy_factor.npz not found")

    output_path = choose_output(OUTPUT_CANDIDATES)
    debug_path = choose_output(DEBUG_CANDIDATES)

    print("=" * 78)
    print("BUILD POLICY DIRECTION CONSISTENCY FACTOR")
    print("=" * 78)
    print("signed daily:", signed_path)
    print("base npz    :", base_path)

    with np.load(base_path) as z:
        dates = z["dates"].astype(np.int64)
        codes = z["codes"].astype(np.int64)

    signed = pd.read_parquet(signed_path)

    required = {"date", "industry"}
    missing = required - set(signed.columns)
    if missing:
        raise KeyError(
            "signed daily missing required columns %s; actual=%s"
            % (sorted(missing), list(signed.columns))
        )

    signed_col = detect_signed_attention_column(signed.columns)
    print("using signed column:", signed_col)

    signed = signed[
        ["date", "industry", signed_col]
    ].copy()

    signed["date_int"] = normalize_date(signed["date"])
    signed["code"] = pd.to_numeric(
        signed["industry"],
        errors="coerce",
    )
    signed["signed_signal"] = pd.to_numeric(
        signed[signed_col],
        errors="coerce",
    ).fillna(0.0)

    signed = signed[
        signed["date_int"].notna()
        & signed["code"].notna()
    ].copy()

    signed["date_int"] = signed["date_int"].astype(np.int64)
    signed["code"] = signed["code"].astype(np.int64)

    valid_dates = set(dates.tolist())
    valid_codes = set(codes.tolist())

    signed = signed[
        signed["date_int"].isin(valid_dates)
        & signed["code"].isin(valid_codes)
    ]

    # In case there are duplicate rows, aggregate signed mass.
    daily = (
        signed.groupby(
            ["date_int", "code"],
            as_index=False,
        )
        .agg(
            signed_signal=("signed_signal", "sum"),
        )
    )

    full_index = pd.MultiIndex.from_product(
        [dates, codes],
        names=["date_int", "code"],
    )

    grid = (
        daily
        .set_index(["date_int", "code"])
        .reindex(full_index)
        .fillna(0.0)
        .reset_index()
    )

    signed_matrix = (
        grid["signed_signal"]
        .values
        .reshape(len(dates), len(codes))
        .astype(np.float64)
    )

    sdf = pd.DataFrame(signed_matrix)

    signed_sum20 = (
        sdf.rolling(
            ROLL_WINDOW,
            min_periods=1,
        ).sum()
    )

    abs_sum20 = (
        sdf.abs()
        .rolling(
            ROLL_WINDOW,
            min_periods=1,
        ).sum()
    )

    consistency20 = (
        signed_sum20.abs()
        / abs_sum20.where(
            abs_sum20 > EPS,
            np.nan,
        )
    ).values.astype(np.float64)

    surprise, rank = historical_surprise_rank(
        consistency20
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        dates=dates,
        codes=codes,
        policy_direction_consistency_surprise_20_rank=rank,
        policy_direction_consistency_20=consistency20,
        policy_direction_consistency_surprise_20=surprise,
        signed_signal=signed_matrix,
    )

    try:
        debug = pd.DataFrame({
            "date": np.repeat(dates, len(codes)),
            "industry": np.tile(codes, len(dates)),
            "signed_signal": signed_matrix.reshape(-1),
            "consistency_20": consistency20.reshape(-1),
            "consistency_surprise_20": surprise.reshape(-1),
            "consistency_rank": rank.reshape(-1),
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
        print("WARNING: debug parquet save failed:", repr(e))

    print("")
    print("grid:", len(dates), "x", len(codes))
    print(
        "finite consistency:",
        "%.4f" % np.isfinite(consistency20).mean()
    )
    print(
        "finite surprise:",
        "%.4f" % np.isfinite(surprise).mean()
    )
    print(
        "finite rank:",
        "%.4f" % np.isfinite(rank).mean()
    )

    finite_c = consistency20[np.isfinite(consistency20)]

    if finite_c.size:
        q = np.quantile(
            finite_c,
            [0.1, 0.25, 0.5, 0.75, 0.9],
        )

        print(
            "consistency q10/q25/q50/q75/q90:",
            " ".join("%.3f" % x for x in q),
        )

    print("")
    print("saved:", output_path)

    with np.load(output_path) as z:
        for k in z.files:
            print(" ", k, z[k].shape, z[k].dtype)

    print("=" * 78)


if __name__ == "__main__":
    main()
