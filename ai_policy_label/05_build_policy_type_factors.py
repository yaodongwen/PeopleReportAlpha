# -*- coding: utf-8 -*-

"""
05_build_policy_type_factors.py

Build Factor32-37 from the ACTUAL merged AI-label file:

    outputs/article_industry_policy.parquet

Confirmed columns include:
    article_id
    date_x
    industry
    framework_code
    article_weight
    confidence
    relevance
    policy_strength
    policy_type
    ...

Also uses:
    outputs/industry_signed_daily.parquet
    ../data/factor/people_daily_ai_policy_factor.npz

Output:
    ../data/factor/people_daily_policy_type_factor.npz

Factor definitions
------------------
32 supportive_surprise_20_rank
   policy_type in:
   support, stimulus, subsidy, tax, investment, demand, reform

33 restrictive_surprise_20_rank
   policy_type in:
   regulation, restriction, risk

34 investment_subsidy_surprise_20_rank
   policy_type in:
   investment, subsidy, tax

35 demand_stimulus_surprise_20_rank
   policy_type in:
   support, stimulus, demand

36 supportive_residual_rank
   Factor32 residualized daily against existing
   quality_surprise_20_rank, then cross-sectionally re-ranked.

37 restrictive_residual_rank
   Factor33 residualized daily against existing
   quality_surprise_20_rank, then cross-sectionally re-ranked.

Timing/standardization follows the existing PeopleDaily factor construction:
    - full natural-calendar grid
    - missing attention filled with 0
    - 20-natural-day rolling sum
    - own-industry historical z-score versus shifted 365-day history
    - min_history = 120
    - daily cross-sectional rank -> approximately [-1, 1]
"""

import os
import numpy as np
import pandas as pd


ARTICLE_POLICY = "outputs/article_industry_policy.parquet"
SIGNED_DAILY = "outputs/industry_signed_daily.parquet"

BASE_FACTOR_NPZ = (
    "../data/factor/"
    "people_daily_ai_policy_factor.npz"
)

OUTPUT_NPZ = (
    "../data/factor/"
    "people_daily_policy_type_factor.npz"
)

DEBUG_DAILY = (
    "outputs/"
    "policy_type_daily_debug.parquet"
)

ROLL_WINDOW = 20
HIST_WINDOW = 365
MIN_HISTORY = 120
STD_FLOOR = 1e-6


SUPPORTIVE_TYPES = set([
    "support",
    "stimulus",
    "subsidy",
    "tax",
    "investment",
    "demand",
    "reform",
])

RESTRICTIVE_TYPES = set([
    "regulation",
    "restriction",
    "risk",
])

INVESTMENT_SUBSIDY_TYPES = set([
    "investment",
    "subsidy",
    "tax",
])

DEMAND_STIMULUS_TYPES = set([
    "support",
    "stimulus",
    "demand",
])


def require_columns(df, cols, name):
    missing = [
        c for c in cols
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            "%s missing columns: %s\nactual columns=%s"
            % (
                name,
                missing,
                list(df.columns),
            )
        )


def normalize_date(s):
    """
    Convert date-like Series to integer YYYYMMDD.
    """

    if np.issubdtype(
        s.dtype,
        np.datetime64
    ):
        return (
            s.dt.strftime("%Y%m%d")
            .astype(np.int64)
        )

    parsed = pd.to_datetime(
        s,
        errors="coerce"
    )

    out = pd.Series(
        np.nan,
        index=s.index,
        dtype=np.float64
    )

    ok = parsed.notna()

    out.loc[ok] = (
        parsed.loc[ok]
        .dt.strftime("%Y%m%d")
        .astype(np.int64)
    )

    return out


def cs_rank_matrix(matrix):
    """
    Daily cross-sectional rank.

    Matches the project convention:
        rank(pct=True) * 2 - 1
    """

    df = pd.DataFrame(
        matrix
    )

    ranked = (
        df.rank(
            axis=1,
            pct=True,
            method="average",
            na_option="keep",
        )
        * 2.0
        - 1.0
    )

    return ranked.values.astype(
        np.float64
    )


def rolling_surprise_rank(attention):
    """
    attention shape:
        [natural calendar date, SW1 industry]

    1. 20-natural-day rolling sum
    2. compare with own-industry shifted 365-day history
    3. z-score
    4. daily cross-sectional rank
    """

    n_dates, n_codes = attention.shape

    rolling_sum = (
        pd.DataFrame(attention)
        .rolling(
            window=ROLL_WINDOW,
            min_periods=1,
        )
        .sum()
        .values
    )

    surprise = np.full(
        (n_dates, n_codes),
        np.nan,
        dtype=np.float64,
    )

    for j in range(n_codes):

        s = pd.Series(
            rolling_sum[:, j]
        )

        # No look-ahead:
        # today's historical benchmark only uses <= yesterday.
        hist = s.shift(1)

        mean = hist.rolling(
            window=HIST_WINDOW,
            min_periods=MIN_HISTORY,
        ).mean()

        std = hist.rolling(
            window=HIST_WINDOW,
            min_periods=MIN_HISTORY,
        ).std()

        std = std.where(
            std > STD_FLOOR,
            np.nan,
        )

        surprise[:, j] = (
            (s - mean) / std
        ).values

    return cs_rank_matrix(
        surprise
    )


def residualize_and_rank(
    y_rank,
    x_rank
):
    """
    Daily cross-sectional regression:

        y = intercept + beta*x + residual

    Then daily cross-sectional re-rank residual.
    """

    n_dates, n_codes = y_rank.shape

    residual = np.full(
        (n_dates, n_codes),
        np.nan,
        dtype=np.float64,
    )

    for i in range(n_dates):

        y = y_rank[i]
        x = x_rank[i]

        valid = (
            np.isfinite(y)
            &
            np.isfinite(x)
        )

        n = int(
            valid.sum()
        )

        if n < 3:
            continue

        X = np.column_stack([
            np.ones(
                n,
                dtype=np.float64
            ),
            x[valid],
        ])

        beta, _, _, _ = (
            np.linalg.lstsq(
                X,
                y[valid],
                rcond=None,
            )
        )

        residual[
            i,
            valid
        ] = (
            y[valid]
            -
            np.dot(
                X,
                beta
            )
        )

    return cs_rank_matrix(
        residual
    )


def print_policy_distribution(df):

    print("")
    print("=" * 80)
    print("POLICY TYPE DISTRIBUTION")
    print("=" * 80)

    counts = (
        df["policy_type_norm"]
        .value_counts(
            dropna=False
        )
    )

    print(
        counts.to_string()
    )

    print("=" * 80)


def main():

    # =========================================================
    # Check input files
    # =========================================================

    for path in [
        ARTICLE_POLICY,
        SIGNED_DAILY,
        BASE_FACTOR_NPZ,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                path
            )

    # =========================================================
    # Load existing factor calendar / industry codes
    # =========================================================

    base = np.load(
        BASE_FACTOR_NPZ
    )

    dates = (
        base["dates"]
        .astype(np.int64)
    )

    codes = (
        base["codes"]
        .astype(np.int64)
    )

    if (
        "quality_surprise_20_rank"
        not in base.files
    ):
        raise KeyError(
            "quality_surprise_20_rank "
            "not found in %s"
            % BASE_FACTOR_NPZ
        )

    quality_rank = (
        base[
            "quality_surprise_20_rank"
        ]
        .astype(np.float64)
    )

    # =========================================================
    # Load ACTUAL merged article × industry × AI-label file
    # =========================================================

    article = pd.read_parquet(
        ARTICLE_POLICY
    )

    require_columns(
        article,
        [
            "article_id",
            "date_x",
            "industry",
            "framework_code",
            "article_weight",
            "confidence",
            "relevance",
            "policy_strength",
            "policy_type",
        ],
        ARTICLE_POLICY
    )

    print("")
    print("=" * 80)
    print("ARTICLE POLICY INPUT")
    print("=" * 80)

    print(
        "rows:",
        len(article)
    )

    print(
        "date:",
        article["date_x"].min(),
        "->",
        article["date_x"].max()
    )

    print(
        "industries:",
        article[
            "framework_code"
        ].nunique()
    )

    # =========================================================
    # Normalize basic fields
    # =========================================================

    article = article.copy()

    article["_date"] = normalize_date(
        article["date_x"]
    )

    article["_code"] = pd.to_numeric(
        article["framework_code"],
        errors="coerce",
    )

    article[
        "policy_type_norm"
    ] = (
        article["policy_type"]
        .fillna("other")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Only successful/usable AI labels contribute AI weight.
    article_weight = pd.to_numeric(
        article["article_weight"],
        errors="coerce",
    ).fillna(0.0)

    confidence = pd.to_numeric(
        article["confidence"],
        errors="coerce",
    ).fillna(0.0)

    relevance = pd.to_numeric(
        article["relevance"],
        errors="coerce",
    ).fillna(0.0)

    policy_strength = pd.to_numeric(
        article["policy_strength"],
        errors="coerce",
    ).fillna(0.0)

    article["ai_weight_calc"] = (
        article_weight
        *
        confidence
        *
        relevance
        *
        policy_strength
    )

    # Keep only valid date/code rows.
    article = article[
        article["_date"].notna()
        &
        article["_code"].notna()
    ].copy()

    article["_date"] = (
        article["_date"]
        .astype(np.int64)
    )

    article["_code"] = (
        article["_code"]
        .astype(np.int64)
    )

    # Only project SW1 codes.
    article = article[
        article["_code"].isin(
            set(codes.tolist())
        )
    ].copy()

    print_policy_distribution(
        article
    )

    usable_ai = (
        (
            pd.to_numeric(
                article["confidence"],
                errors="coerce",
            ).notna()
        )
        &
        (
            pd.to_numeric(
                article["relevance"],
                errors="coerce",
            ).notna()
        )
        &
        (
            pd.to_numeric(
                article["policy_strength"],
                errors="coerce",
            ).notna()
        )
    )

    print("")
    print(
        "usable AI rows:",
        int(
            usable_ai.sum()
        )
    )

    print(
        "missing/failed AI rows:",
        int(
            (~usable_ai).sum()
        )
    )

    # =========================================================
    # Type-specific AI weights
    # =========================================================

    article[
        "w_supportive"
    ] = np.where(
        article[
            "policy_type_norm"
        ].isin(
            SUPPORTIVE_TYPES
        ),
        article[
            "ai_weight_calc"
        ],
        0.0,
    )

    article[
        "w_restrictive"
    ] = np.where(
        article[
            "policy_type_norm"
        ].isin(
            RESTRICTIVE_TYPES
        ),
        article[
            "ai_weight_calc"
        ],
        0.0,
    )

    article[
        "w_investment_subsidy"
    ] = np.where(
        article[
            "policy_type_norm"
        ].isin(
            INVESTMENT_SUBSIDY_TYPES
        ),
        article[
            "ai_weight_calc"
        ],
        0.0,
    )

    article[
        "w_demand_stimulus"
    ] = np.where(
        article[
            "policy_type_norm"
        ].isin(
            DEMAND_STIMULUS_TYPES
        ),
        article[
            "ai_weight_calc"
        ],
        0.0,
    )

    # =========================================================
    # Aggregate AI weights by natural date × SW1 industry
    # =========================================================

    type_daily = (
        article.groupby(
            [
                "_date",
                "_code",
            ],
            as_index=False,
        )
        .agg(
            total_ai_weight=(
                "ai_weight_calc",
                "sum",
            ),
            supportive_weight=(
                "w_supportive",
                "sum",
            ),
            restrictive_weight=(
                "w_restrictive",
                "sum",
            ),
            investment_subsidy_weight=(
                "w_investment_subsidy",
                "sum",
            ),
            demand_stimulus_weight=(
                "w_demand_stimulus",
                "sum",
            ),
        )
    )

    # =========================================================
    # Load existing signed daily.
    #
    # We use its quality_attention as the already-established
    # normalization to total PeopleDaily articles on that day.
    #
    # Then split quality_attention across policy types according
    # to each type's share of AI weight within date × industry.
    #
    # This keeps Factor32-37 on the same attention scale as the
    # existing Factor8 / Factor13 / Factor22 pipeline.
    # =========================================================

    signed = pd.read_parquet(
        SIGNED_DAILY
    ).copy()

    if "quality_attention" not in signed.columns:
        raise KeyError(
            "quality_attention not found "
            "in %s; columns=%s"
            % (
                SIGNED_DAILY,
                list(
                    signed.columns
                ),
            )
        )

    # ---------------------------------------------------------
    # Resolve signed-daily industry code.
    #
    # In the actual project file, industry_signed_daily.parquet
    # contains the Chinese industry name in column "industry",
    # but no framework_code.  Build a deterministic mapping from
    # the already-loaded article_industry_policy.parquet:
    #
    #     industry -> framework_code
    #
    # and merge it into signed daily.
    # ---------------------------------------------------------

    if "framework_code" in signed.columns:

        signed["_code"] = pd.to_numeric(
            signed["framework_code"],
            errors="coerce",
        )

    elif "industry_code" in signed.columns:

        signed["_code"] = pd.to_numeric(
            signed["industry_code"],
            errors="coerce",
        )

    elif "code" in signed.columns:

        signed["_code"] = pd.to_numeric(
            signed["code"],
            errors="coerce",
        )

    elif "industry" in signed.columns:

        # In the actual industry_signed_daily.parquet, column
        # "industry" already stores the 6-digit SW1 framework_code
        # as text, e.g. "110000", "420000", ...
        #
        # So do NOT map by Chinese industry name here.
        signed["_code"] = pd.to_numeric(
            signed["industry"],
            errors="coerce",
        )

        print("")
        print("Signed-daily industry codes:")
        print(
            "  rows:",
            len(signed)
        )
        print(
            "  numeric codes:",
            int(
                signed["_code"].notna().sum()
            )
        )
        print(
            "  invalid:",
            int(
                signed["_code"].isna().sum()
            )
        )

        if signed["_code"].isna().any():

            bad_values = (
                signed.loc[
                    signed["_code"].isna(),
                    "industry",
                ]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            raise RuntimeError(
                "Non-numeric values found in signed daily industry column: %s"
                % bad_values
            )

    else:

        raise KeyError(
            "Cannot find industry identifier "
            "in %s; columns=%s"
            % (
                SIGNED_DAILY,
                list(
                    signed.columns
                ),
            )
        )

    if "date" in signed.columns:
        signed_date_col = "date"
    elif "news_date" in signed.columns:
        signed_date_col = "news_date"
    elif "publish_date" in signed.columns:
        signed_date_col = (
            "publish_date"
        )
    else:
        raise KeyError(
            "Cannot find date column "
            "in %s; columns=%s"
            % (
                SIGNED_DAILY,
                list(
                    signed.columns
                ),
            )
        )

    signed["_date"] = normalize_date(
        signed[
            signed_date_col
        ]
    )

    # _code has already been resolved above.
    signed["_code"] = pd.to_numeric(
        signed["_code"],
        errors="coerce",
    )

    signed = signed[
        signed["_date"].notna()
        &
        signed["_code"].notna()
    ].copy()

    signed["_date"] = (
        signed["_date"]
        .astype(np.int64)
    )

    signed["_code"] = (
        signed["_code"]
        .astype(np.int64)
    )

    signed = signed[
        signed["_code"].isin(
            set(codes.tolist())
        )
    ].copy()

    daily = (
        signed[
            [
                "_date",
                "_code",
                "quality_attention",
            ]
        ]
        .merge(
            type_daily,
            on=[
                "_date",
                "_code",
            ],
            how="left",
        )
    )

    for col in [
        "total_ai_weight",
        "supportive_weight",
        "restrictive_weight",
        "investment_subsidy_weight",
        "demand_stimulus_weight",
    ]:
        daily[col] = (
            daily[col]
            .fillna(0.0)
        )

    total = daily[
        "total_ai_weight"
    ].values.astype(
        np.float64
    )

    quality = daily[
        "quality_attention"
    ].values.astype(
        np.float64
    )

    def allocate(weight_col):

        w = daily[
            weight_col
        ].values.astype(
            np.float64
        )

        out = np.zeros(
            len(daily),
            dtype=np.float64,
        )

        valid = (
            np.isfinite(total)
            &
            np.isfinite(quality)
            &
            (total > 0.0)
        )

        out[valid] = (
            quality[valid]
            *
            w[valid]
            /
            total[valid]
        )

        return out

    daily[
        "supportive_attention"
    ] = allocate(
        "supportive_weight"
    )

    daily[
        "restrictive_attention"
    ] = allocate(
        "restrictive_weight"
    )

    daily[
        "investment_subsidy_attention"
    ] = allocate(
        "investment_subsidy_weight"
    )

    daily[
        "demand_stimulus_attention"
    ] = allocate(
        "demand_stimulus_weight"
    )

    # =========================================================
    # Full natural-calendar × 30-industry grid
    # =========================================================

    full_index = (
        pd.MultiIndex
        .from_product(
            [
                dates,
                codes,
            ],
            names=[
                "_date",
                "_code",
            ],
        )
    )

    full = (
        daily[
            [
                "_date",
                "_code",
                "supportive_attention",
                "restrictive_attention",
                "investment_subsidy_attention",
                "demand_stimulus_attention",
            ]
        ]
        .set_index(
            [
                "_date",
                "_code",
            ]
        )
        .reindex(
            full_index
        )
        .fillna(0.0)
        .reset_index()
    )

    def matrix(col):

        p = full.pivot(
            index="_date",
            columns="_code",
            values=col,
        )

        p = p.reindex(
            index=dates,
            columns=codes,
        )

        return p.values.astype(
            np.float64
        )

    supportive_attention = matrix(
        "supportive_attention"
    )

    restrictive_attention = matrix(
        "restrictive_attention"
    )

    invest_sub_attention = matrix(
        "investment_subsidy_attention"
    )

    demand_attention = matrix(
        "demand_stimulus_attention"
    )

    # =========================================================
    # Build Factor32-35
    # =========================================================

    print("")
    print("Building Factor32 supportive...")

    f32 = rolling_surprise_rank(
        supportive_attention
    )

    print(
        "Building Factor33 restrictive..."
    )

    f33 = rolling_surprise_rank(
        restrictive_attention
    )

    print(
        "Building Factor34 "
        "investment/subsidy/tax..."
    )

    f34 = rolling_surprise_rank(
        invest_sub_attention
    )

    print(
        "Building Factor35 "
        "support/stimulus/demand..."
    )

    f35 = rolling_surprise_rank(
        demand_attention
    )

    # =========================================================
    # Build orthogonalized Factor36-37
    # =========================================================

    print(
        "Building Factor36 "
        "supportive residual..."
    )

    f36 = residualize_and_rank(
        f32,
        quality_rank,
    )

    print(
        "Building Factor37 "
        "restrictive residual..."
    )

    f37 = residualize_and_rank(
        f33,
        quality_rank,
    )

    # =========================================================
    # Save
    # =========================================================

    output_dir = os.path.dirname(
        OUTPUT_NPZ
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    np.savez_compressed(
        OUTPUT_NPZ,

        dates=dates,
        codes=codes,

        supportive_surprise_20_rank=f32,

        restrictive_surprise_20_rank=f33,

        investment_subsidy_surprise_20_rank=f34,

        demand_stimulus_surprise_20_rank=f35,

        supportive_residual_rank=f36,

        restrictive_residual_rank=f37,
    )

    # Debug file is useful for auditing policy-type attention.
    daily[
        [
            "_date",
            "_code",
            "quality_attention",
            "total_ai_weight",
            "supportive_attention",
            "restrictive_attention",
            "investment_subsidy_attention",
            "demand_stimulus_attention",
        ]
    ].to_parquet(
        DEBUG_DAILY,
        index=False,
    )

    # =========================================================
    # Diagnostics
    # =========================================================

    print("")
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        "output:",
        OUTPUT_NPZ
    )

    print(
        "shape:",
        f32.shape
    )

    print(
        "dates:",
        int(dates[0]),
        "->",
        int(dates[-1])
    )

    print(
        "codes:",
        len(codes)
    )

    print("")
    print(
        "Factor32 supportive_surprise_20_rank"
    )

    print(
        "Factor33 restrictive_surprise_20_rank"
    )

    print(
        "Factor34 investment_subsidy_surprise_20_rank"
    )

    print(
        "Factor35 demand_stimulus_surprise_20_rank"
    )

    print(
        "Factor36 supportive_residual_rank"
    )

    print(
        "Factor37 restrictive_residual_rank"
    )

    print("")
    print(
        "debug:",
        DEBUG_DAILY
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
