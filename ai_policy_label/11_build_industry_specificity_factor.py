#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
11_build_industry_specificity_factor.py

Factor77: policy industry specificity

Idea
----
For each PeopleDaily article:
    specificity = 1 / number_of_distinct_SW1_industries_linked_to_article

Then for each article-industry row:
    specific_weight = ai_weight * specificity

For each natural calendar day × industry:
    daily_specific_attention
        = sum(specific_weight) / total_articles_that_day

Then:
    rolling 20-natural-day mean
    -> shifted 365-natural-day own-industry historical surprise
       min_history = 120
    -> daily cross-sectional rank in [-1, 1]

Timing:
    NPZ is indexed by natural calendar day t.
    Runtime trades on T using t = T-1, same as existing factors.

Important:
    This factor does NOT use price in construction.
    Runtime applies the same acceleration long-side veto as the current
    PeopleDaily factor family.
"""

import os
import numpy as np
import pandas as pd


ARTICLE_CANDIDATES = [
    "ai_policy_label/outputs/article_industry_policy.parquet",
    "outputs/article_industry_policy.parquet",
]

SIGNED_DAILY_CANDIDATES = [
    "ai_policy_label/outputs/industry_signed_daily.parquet",
    "outputs/industry_signed_daily.parquet",
]

BASE_CANDIDATES = [
    "data/factor/people_daily_ai_policy_factor.npz",
    "../data/factor/people_daily_ai_policy_factor.npz",
]

OUTPUT_CANDIDATES = [
    "data/factor/people_daily_industry_specificity_factor.npz",
    "../data/factor/people_daily_industry_specificity_factor.npz",
]

DEBUG_CANDIDATES = [
    "ai_policy_label/outputs/industry_specificity_daily_debug.parquet",
    "outputs/industry_specificity_daily_debug.parquet",
]

ROLL_WINDOW = 20
HIST_WINDOW = 365
MIN_HISTORY = 120
STD_FLOOR = 1e-6

ARTICLE_ID_CANDIDATES = [
    "article_id",
    "news_id",
    "doc_id",
    "id",
    "source_id",
]

DATE_CANDIDATES = [
    "date",
    "date_x",
    "date_y",
    "publish_date",
    "published_date",
    "pub_date",
]

WEIGHT_CANDIDATES = [
    "ai_weight",
]

CODE_CANDIDATES = [
    "framework_code",
    "industry_code",
    "code",
]


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


def first_col(columns, candidates, label):
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(
        "%s column not found. candidates=%s actual=%s"
        % (label, candidates, list(columns))
    )


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


def detect_article_key(df):
    for c in ARTICLE_ID_CANDIDATES:
        if c in df.columns:
            print("article key:", c)
            return df[c].astype(str)

    # Fallback: date + title is much safer than row index because
    # article_industry_policy has one article repeated across industries.
    if "title" in df.columns:
        date_col = first_col(
            df.columns,
            DATE_CANDIDATES,
            "date",
        )
        print("article key: fallback date + title")
        return (
            df[date_col].astype(str)
            + "||"
            + df["title"].fillna("").astype(str)
        )

    raise KeyError(
        "Cannot identify article identity. Need one of %s or title+date. actual=%s"
        % (ARTICLE_ID_CANDIDATES, list(df.columns))
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


def build_total_articles_by_date(signed_daily, date_col):
    # Prefer explicit denominator if already stored by the signed-daily builder.
    explicit_candidates = [
        "total_articles",
        "daily_total_articles",
        "n_articles",
        "article_count",
    ]

    for c in explicit_candidates:
        if c in signed_daily.columns:
            tmp = signed_daily[[date_col, c]].copy()
            tmp["date_int"] = normalize_date(tmp[date_col])
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

            denom = (
                tmp.dropna(subset=["date_int", c])
                .groupby("date_int")[c]
                .max()
            )

            print("daily denominator column:", c)
            return denom

    raise KeyError(
        "No total-article denominator column found in signed daily. "
        "Expected one of %s; actual=%s"
        % (explicit_candidates, list(signed_daily.columns))
    )


def main():
    article_path = first_existing(ARTICLE_CANDIDATES)
    signed_daily_path = first_existing(SIGNED_DAILY_CANDIDATES)
    base_path = first_existing(BASE_CANDIDATES)

    if article_path is None:
        raise FileNotFoundError("article_industry_policy.parquet not found")
    if signed_daily_path is None:
        raise FileNotFoundError("industry_signed_daily.parquet not found")
    if base_path is None:
        raise FileNotFoundError("people_daily_ai_policy_factor.npz not found")

    output_path = choose_output(OUTPUT_CANDIDATES)
    debug_path = choose_output(DEBUG_CANDIDATES)

    print("=" * 80)
    print("BUILD INDUSTRY SPECIFICITY FACTOR")
    print("=" * 80)
    print("article file :", article_path)
    print("signed daily :", signed_daily_path)
    print("base npz     :", base_path)

    with np.load(base_path) as z:
        dates = z["dates"].astype(np.int64)
        codes = z["codes"].astype(np.int64)

    article = pd.read_parquet(article_path)
    signed_daily = pd.read_parquet(signed_daily_path)

    date_col = first_col(
        article.columns,
        DATE_CANDIDATES,
        "date",
    )
    code_col = first_col(
        article.columns,
        CODE_CANDIDATES,
        "industry code",
    )
    weight_col = None
    for c in WEIGHT_CANDIDATES:
        if c in article.columns:
            weight_col = c
            break

    signed_date_col = first_col(
        signed_daily.columns,
        DATE_CANDIDATES,
        "signed daily date",
    )

    print("article date column:", date_col)
    print("industry code column:", code_col)
    print("signed daily date column:", signed_date_col)

    article = article.copy()
    article["_article_key"] = detect_article_key(article)
    article["_date_int"] = normalize_date(article[date_col])
    article["_code"] = pd.to_numeric(
        article[code_col],
        errors="coerce",
    )
    if weight_col is not None:
        print("ai weight column:", weight_col)

        article["_ai_weight"] = pd.to_numeric(
            article[weight_col],
            errors="coerce",
        )

    else:
        required_weight_parts = [
            "article_weight",
            "confidence",
            "relevance",
            "policy_strength",
        ]

        missing_weight_parts = [
            c for c in required_weight_parts
            if c not in article.columns
        ]

        if missing_weight_parts:
            raise KeyError(
                "ai_weight not found and cannot derive it; "
                "missing columns=%s actual=%s"
                % (
                    missing_weight_parts,
                    list(article.columns),
                )
            )

        print(
            "ai weight column: derived as "
            "article_weight * confidence * relevance * policy_strength"
        )

        article["_ai_weight"] = (
            pd.to_numeric(
                article["article_weight"],
                errors="coerce",
            )
            * pd.to_numeric(
                article["confidence"],
                errors="coerce",
            )
            * pd.to_numeric(
                article["relevance"],
                errors="coerce",
            )
            * pd.to_numeric(
                article["policy_strength"],
                errors="coerce",
            )
        )

    article = article[
        article["_date_int"].notna()
        & article["_code"].notna()
        & article["_ai_weight"].notna()
    ].copy()

    article["_date_int"] = article["_date_int"].astype(np.int64)
    article["_code"] = article["_code"].astype(np.int64)

    valid_dates = set(dates.tolist())
    valid_codes = set(codes.tolist())

    article = article[
        article["_date_int"].isin(valid_dates)
        & article["_code"].isin(valid_codes)
    ].copy()

    # Count distinct industries for each article.
    n_industries = (
        article.groupby("_article_key")["_code"]
        .nunique()
        .rename("_n_industries")
    )

    article = article.join(
        n_industries,
        on="_article_key",
    )

    article["_specificity"] = (
        1.0 / article["_n_industries"].astype(np.float64)
    )

    article["_specific_weight"] = (
        article["_ai_weight"]
        * article["_specificity"]
    )

    # Daily denominator = ALL PeopleDaily articles that natural day,
    # consistent with the existing signed-daily construction.
    denom = build_total_articles_by_date(
        signed_daily,
        signed_date_col,
    )

    numerator = (
        article.groupby(
            ["_date_int", "_code"],
            as_index=False,
        )
        .agg(
            specific_weight=("_specific_weight", "sum"),
            ai_weight=("_ai_weight", "sum"),
            mean_specificity=("_specificity", "mean"),
            linked_articles=("_article_key", "nunique"),
        )
    )

    numerator["total_articles"] = numerator["_date_int"].map(denom)

    numerator["daily_specific_attention"] = (
        numerator["specific_weight"]
        / numerator["total_articles"].where(
            numerator["total_articles"] > 0,
            np.nan,
        )
    )

    full_index = pd.MultiIndex.from_product(
        [dates, codes],
        names=["date_int", "code"],
    )

    daily = (
        numerator.rename(
            columns={
                "_date_int": "date_int",
                "_code": "code",
            }
        )
        .set_index(["date_int", "code"])
        .reindex(full_index)
    )

    # Missing industry-day means no specific policy attention that day.
    daily_attention = (
        daily["daily_specific_attention"]
        .fillna(0.0)
        .values
        .reshape(len(dates), len(codes))
        .astype(np.float64)
    )

    rolling20 = (
        pd.DataFrame(daily_attention)
        .rolling(
            ROLL_WINDOW,
            min_periods=1,
        )
        .mean()
        .values
        .astype(np.float64)
    )

    surprise, rank = historical_surprise_rank(
        rolling20
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        dates=dates,
        codes=codes,
        industry_specificity_attention_20=rolling20,
        industry_specificity_surprise_20=surprise,
        industry_specificity_surprise_20_rank=rank,
        daily_industry_specificity_attention=daily_attention,
    )

    try:
        debug = pd.DataFrame({
            "date": np.repeat(dates, len(codes)),
            "industry": np.tile(codes, len(dates)),
            "daily_specific_attention": daily_attention.reshape(-1),
            "specific_attention_20": rolling20.reshape(-1),
            "specificity_surprise_20": surprise.reshape(-1),
            "specificity_rank": rank.reshape(-1),
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
    print("usable article-industry rows:", len(article))
    print(
        "unique articles:",
        article["_article_key"].nunique(),
    )

    nind = article.drop_duplicates(
        "_article_key"
    )["_n_industries"]

    if len(nind):
        print(
            "article industry-count q50/q75/q90/q95/max:",
            "%.1f %.1f %.1f %.1f %d"
            % (
                nind.quantile(0.50),
                nind.quantile(0.75),
                nind.quantile(0.90),
                nind.quantile(0.95),
                int(nind.max()),
            )
        )

    print(
        "finite surprise:",
        "%.4f" % np.isfinite(surprise).mean(),
    )
    print(
        "finite rank:",
        "%.4f" % np.isfinite(rank).mean(),
    )
    print("")
    print("saved:", output_path)

    with np.load(output_path) as z:
        for k in z.files:
            print(" ", k, z[k].shape, z[k].dtype)

    print("=" * 80)


if __name__ == "__main__":
    main()
