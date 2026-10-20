#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_build_concentration_novelty_factors.py

Build two new PeopleDaily text-structure factors.

Factor56: policy_concentration_surprise_20_rank
------------------------------------------------
For each industry and natural calendar day, use quality_attention.

concentration_20
    = max(quality_attention over last 20 natural days)
      / sum(quality_attention over last 20 natural days)

Interpretation:
    Same 20-day total policy attention can come from persistent reporting
    or from one/few highly concentrated reporting days.

Then:
    shifted 365-natural-day own-industry historical z-score
    min_history=120
    daily cross-sectional rank.

Factor57: policy_first_appearance_surprise_20_rank
---------------------------------------------------
This is the requested "某种报道第一次出现" dimension.

At article-industry level, define a report category by AI policy_type.
For each (industry, policy_type), an article is marked FIRST_APPEARANCE_365 if:
    - this policy_type has never appeared before for that industry, OR
    - the previous occurrence was more than 365 natural days ago.

This is deliberately a "first / reappearance after long absence" signal.
Strict first-ever is also saved for diagnostics, but is too sparse to use
as the main factor over a 16-year sample.

Daily first-appearance attention:
    sum(ai_weight of FIRST_APPEARANCE_365 rows) / total_articles_that_day

where:
    ai_weight = article_weight * confidence * relevance * policy_strength

Then build:
    first_appearance_share_20
      = rolling20 sum(first_appearance_attention)
        / rolling20 sum(quality_attention)

Finally:
    shifted 365-day own-industry historical z-score
    -> daily CS rank.

Timing:
    NPZ values live on natural calendar day t.
    Alpha runtime trades on T using calendar T-1, same as existing factors.
"""

import os
import numpy as np
import pandas as pd


SIGNED_DAILY_CANDIDATES = [
    "ai_policy_label/outputs/industry_signed_daily.parquet",
    "outputs/industry_signed_daily.parquet",
]

ARTICLE_CANDIDATES = [
    "ai_policy_label/outputs/article_industry_policy.parquet",
    "outputs/article_industry_policy.parquet",
]

BASE_FACTOR_CANDIDATES = [
    "data/factor/people_daily_ai_policy_factor.npz",
    "../data/factor/people_daily_ai_policy_factor.npz",
]

OUTPUT_CANDIDATES = [
    "data/factor/people_daily_concentration_novelty_factor.npz",
    "../data/factor/people_daily_concentration_novelty_factor.npz",
]

DEBUG_CANDIDATES = [
    "ai_policy_label/outputs/concentration_novelty_daily_debug.parquet",
    "outputs/concentration_novelty_daily_debug.parquet",
]

WINDOW = 20
HIST_WINDOW = 365
MIN_HISTORY = 120
NOVELTY_LOOKBACK_DAYS = 365
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
        if parent and os.path.isdir(parent):
            return p
    return paths[0]


def normalize_date(s):
    if np.issubdtype(s.dtype, np.datetime64):
        return s.dt.strftime("%Y%m%d").astype(np.int64)

    numeric = pd.to_numeric(s, errors="coerce")
    good = numeric.notna()

    if good.any():
        vals = numeric.loc[good].astype(np.int64)
        if ((vals >= 19000101) & (vals <= 21001231)).all():
            out = pd.Series(np.nan, index=s.index, dtype=np.float64)
            out.loc[good] = vals.values
            return out

    parsed = pd.to_datetime(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=np.float64)
    ok = parsed.notna()
    out.loc[ok] = (
        parsed.loc[ok]
        .dt.strftime("%Y%m%d")
        .astype(np.int64)
    )
    return out


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


def historical_surprise_rank(raw_signal):
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

    return surprise, cs_rank_matrix(surprise)


def detect_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            "None of columns %s found. Actual columns=%s"
            % (candidates, list(df.columns))
        )
    return None


def main():
    signed_path = first_existing(SIGNED_DAILY_CANDIDATES)
    article_path = first_existing(ARTICLE_CANDIDATES)
    base_path = first_existing(BASE_FACTOR_CANDIDATES)

    if signed_path is None:
        raise FileNotFoundError("industry_signed_daily.parquet not found")
    if article_path is None:
        raise FileNotFoundError("article_industry_policy.parquet not found")
    if base_path is None:
        raise FileNotFoundError("people_daily_ai_policy_factor.npz not found")

    output_path = choose_output(OUTPUT_CANDIDATES)
    debug_path = choose_output(DEBUG_CANDIDATES)

    print("=" * 78)
    print("BUILD CONCENTRATION + FIRST-APPEARANCE NOVELTY FACTORS")
    print("=" * 78)
    print("signed daily:", signed_path)
    print("article file:", article_path)
    print("base npz    :", base_path)
    print("output npz  :", output_path)

    with np.load(base_path) as base:
        dates = base["dates"].astype(np.int64)
        codes = base["codes"].astype(np.int64)

    date_to_pos = {int(d): i for i, d in enumerate(dates)}
    code_to_pos = {int(c): i for i, c in enumerate(codes)}

    # ---------------------------------------------------------
    # Signed daily full grid
    # ---------------------------------------------------------
    signed = pd.read_parquet(signed_path)

    required = {"date", "industry", "quality_attention", "total_articles"}
    missing = required - set(signed.columns)
    if missing:
        raise KeyError(
            "signed daily missing %s; actual=%s"
            % (sorted(missing), list(signed.columns))
        )

    signed = signed[
        ["date", "industry", "quality_attention", "total_articles"]
    ].copy()

    signed["date_int"] = normalize_date(signed["date"])
    signed["code"] = pd.to_numeric(
        signed["industry"],
        errors="coerce",
    )

    signed["quality_attention"] = pd.to_numeric(
        signed["quality_attention"],
        errors="coerce",
    ).fillna(0.0)

    signed["total_articles"] = pd.to_numeric(
        signed["total_articles"],
        errors="coerce",
    ).fillna(0.0)

    signed = signed[
        signed["date_int"].notna()
        & signed["code"].notna()
    ].copy()

    signed["date_int"] = signed["date_int"].astype(np.int64)
    signed["code"] = signed["code"].astype(np.int64)

    signed = signed[
        signed["date_int"].isin(date_to_pos)
        & signed["code"].isin(code_to_pos)
    ]

    daily = (
        signed.groupby(["date_int", "code"], as_index=False)
        .agg(
            quality_attention=("quality_attention", "sum"),
            total_articles=("total_articles", "max"),
        )
    )

    full_index = pd.MultiIndex.from_product(
        [dates, codes],
        names=["date_int", "code"],
    )

    grid = (
        daily.set_index(["date_int", "code"])
        .reindex(full_index)
        .reset_index()
    )

    grid["quality_attention"] = (
        grid["quality_attention"].fillna(0.0)
    )

    # total_articles is a day-level denominator repeated by industry.
    # For days absent in signed_daily, use zero.
    grid["total_articles"] = (
        grid["total_articles"].fillna(0.0)
    )

    quality = (
        grid["quality_attention"]
        .values
        .reshape(len(dates), len(codes))
        .astype(np.float64)
    )

    total_articles_matrix = (
        grid["total_articles"]
        .values
        .reshape(len(dates), len(codes))
        .astype(np.float64)
    )

    # ---------------------------------------------------------
    # Factor56: concentration_20
    # ---------------------------------------------------------
    qdf = pd.DataFrame(quality)

    roll_sum20 = qdf.rolling(
        WINDOW,
        min_periods=1,
    ).sum()

    roll_max20 = qdf.rolling(
        WINDOW,
        min_periods=1,
    ).max()

    concentration20 = (
        roll_max20
        / roll_sum20.where(roll_sum20 > EPS, np.nan)
    ).values.astype(np.float64)

    conc_surprise, conc_rank = historical_surprise_rank(
        concentration20
    )

    # ---------------------------------------------------------
    # Article-level novelty / first appearance
    # ---------------------------------------------------------
    art = pd.read_parquet(article_path)

    date_col = detect_col(
        art,
        ["date_x", "date", "date_y"],
    )
    code_col = detect_col(
        art,
        ["framework_code", "index_code"],
    )
    type_col = detect_col(
        art,
        ["policy_type"],
    )
    article_weight_col = detect_col(
        art,
        ["article_weight"],
    )
    confidence_col = detect_col(
        art,
        ["confidence"],
    )
    relevance_col = detect_col(
        art,
        ["relevance"],
    )
    strength_col = detect_col(
        art,
        ["policy_strength"],
    )
    ok_col = detect_col(
        art,
        ["ok"],
        required=False,
    )

    cols = [
        date_col,
        code_col,
        type_col,
        article_weight_col,
        confidence_col,
        relevance_col,
        strength_col,
    ]
    if ok_col is not None:
        cols.append(ok_col)

    art = art[cols].copy()

    art["date_int"] = normalize_date(art[date_col])
    art["code"] = pd.to_numeric(
        art[code_col],
        errors="coerce",
    )
    art["policy_type_clean"] = (
        art[type_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if ok_col is not None:
        ok_values = art[ok_col]
        # robustly support bool / 0-1 / string.
        if ok_values.dtype == bool:
            art = art[ok_values].copy()
        else:
            ok_norm = ok_values.astype(str).str.lower()
            art = art[
                ok_norm.isin(["true", "1", "1.0"])
            ].copy()

    for c in [
        article_weight_col,
        confidence_col,
        relevance_col,
        strength_col,
    ]:
        art[c] = pd.to_numeric(
            art[c],
            errors="coerce",
        )

    art = art[
        art["date_int"].notna()
        & art["code"].notna()
        & art["policy_type_clean"].notna()
        & (art["policy_type_clean"] != "")
        & (art["policy_type_clean"] != "nan")
    ].copy()

    art["date_int"] = art["date_int"].astype(np.int64)
    art["code"] = art["code"].astype(np.int64)

    art = art[
        art["date_int"].isin(date_to_pos)
        & art["code"].isin(code_to_pos)
    ].copy()

    art["ai_weight"] = (
        art[article_weight_col].fillna(0.0)
        * art[confidence_col].fillna(0.0)
        * art[relevance_col].fillna(0.0)
        * art[strength_col].fillna(0.0)
    )

    # Convert YYYYMMDD -> datetime ONLY for date-difference calculation.
    art["date_dt"] = pd.to_datetime(
        art["date_int"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    art = art.sort_values(
        ["code", "policy_type_clean", "date_dt"]
    ).reset_index(drop=True)

    prev_date = (
        art.groupby(
            ["code", "policy_type_clean"],
            sort=False,
        )["date_dt"]
        .shift(1)
    )

    gap_days = (
        art["date_dt"] - prev_date
    ).dt.days

    art["is_first_ever"] = prev_date.isna()

    art["is_first_appearance_365"] = (
        prev_date.isna()
        | (gap_days > NOVELTY_LOOKBACK_DAYS)
    )

    art["first_ever_weight"] = (
        art["ai_weight"]
        * art["is_first_ever"].astype(np.float64)
    )

    art["first_appearance_365_weight"] = (
        art["ai_weight"]
        * art["is_first_appearance_365"].astype(np.float64)
    )

    novelty_daily = (
        art.groupby(["date_int", "code"], as_index=False)
        .agg(
            first_appearance_365_weight=(
                "first_appearance_365_weight",
                "sum",
            ),
            first_ever_weight=(
                "first_ever_weight",
                "sum",
            ),
            ai_weight_all=("ai_weight", "sum"),
            first_appearance_rows=(
                "is_first_appearance_365",
                "sum",
            ),
            first_ever_rows=(
                "is_first_ever",
                "sum",
            ),
        )
    )

    nov_grid = (
        novelty_daily
        .set_index(["date_int", "code"])
        .reindex(full_index)
        .fillna(0.0)
        .reset_index()
    )

    first365_weight = (
        nov_grid["first_appearance_365_weight"]
        .values
        .reshape(len(dates), len(codes))
        .astype(np.float64)
    )

    firstever_weight = (
        nov_grid["first_ever_weight"]
        .values
        .reshape(len(dates), len(codes))
        .astype(np.float64)
    )

    # Same denominator convention as quality_attention:
    # weighted article mass / total PeopleDaily articles that day.
    denom = total_articles_matrix.copy()

    first365_attention = np.divide(
        first365_weight,
        denom,
        out=np.zeros_like(first365_weight),
        where=denom > 0,
    )

    firstever_attention = np.divide(
        firstever_weight,
        denom,
        out=np.zeros_like(firstever_weight),
        where=denom > 0,
    )

    # 20-day novelty share relative to total quality attention.
    first365_roll20 = (
        pd.DataFrame(first365_attention)
        .rolling(WINDOW, min_periods=1)
        .sum()
    )

    quality_roll20 = (
        pd.DataFrame(quality)
        .rolling(WINDOW, min_periods=1)
        .sum()
    )

    first_share20 = (
        first365_roll20
        / quality_roll20.where(quality_roll20 > EPS, np.nan)
    ).values.astype(np.float64)

    first_surprise, first_rank = historical_surprise_rank(
        first_share20
    )

    # Strict first-ever version saved only as diagnostic.
    firstever_roll20 = (
        pd.DataFrame(firstever_attention)
        .rolling(WINDOW, min_periods=1)
        .sum()
    )

    firstever_share20 = (
        firstever_roll20
        / quality_roll20.where(quality_roll20 > EPS, np.nan)
    ).values.astype(np.float64)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        dates=dates,
        codes=codes,

        policy_concentration_surprise_20_rank=conc_rank,
        policy_first_appearance_surprise_20_rank=first_rank,

        concentration_20=concentration20,
        policy_concentration_surprise_20=conc_surprise,

        first_appearance_365_attention=first365_attention,
        first_appearance_share_20=first_share20,
        policy_first_appearance_surprise_20=first_surprise,

        # diagnostics
        strict_first_ever_attention=firstever_attention,
        strict_first_ever_share_20=firstever_share20,
    )

    # Debug long form
    try:
        dbg = pd.DataFrame({
            "date": np.repeat(dates, len(codes)),
            "industry": np.tile(codes, len(dates)),
            "quality_attention": quality.reshape(-1),
            "concentration_20": concentration20.reshape(-1),
            "concentration_surprise": conc_surprise.reshape(-1),
            "concentration_rank": conc_rank.reshape(-1),
            "first_appearance_365_attention": first365_attention.reshape(-1),
            "first_appearance_share_20": first_share20.reshape(-1),
            "first_appearance_surprise": first_surprise.reshape(-1),
            "first_appearance_rank": first_rank.reshape(-1),
            "strict_first_ever_attention": firstever_attention.reshape(-1),
            "strict_first_ever_share_20": firstever_share20.reshape(-1),
        })

        os.makedirs(
            os.path.dirname(debug_path),
            exist_ok=True,
        )

        dbg.to_parquet(
            debug_path,
            index=False,
        )

        print("debug parquet:", debug_path)

    except Exception as e:
        print("WARNING: debug parquet failed:", repr(e))

    print("")
    print("article-level novelty diagnostics:")
    print("  usable rows:", len(art))
    print(
        "  first-ever rows:",
        int(art["is_first_ever"].sum())
    )
    print(
        "  first/reappearance >365d rows:",
        int(art["is_first_appearance_365"].sum())
    )

    print("")
    print("finite ratios:")
    print(
        "  concentration:",
        "%.4f" % np.isfinite(conc_rank).mean()
    )
    print(
        "  first appearance:",
        "%.4f" % np.isfinite(first_rank).mean()
    )

    print("")
    print("saved:", output_path)
    with np.load(output_path) as z:
        for k in z.files:
            print(" ", k, z[k].shape, z[k].dtype)

    print("=" * 78)


if __name__ == "__main__":
    main()
