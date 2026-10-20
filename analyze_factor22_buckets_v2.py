# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


INPUT = (
    "outputs/"
    "factor22_bucket_raw.csv"
)


# ============================================================
# t-stat
# ============================================================

def t_stat(x):

    x = pd.Series(x).dropna()

    n = len(x)

    if n < 2:
        return np.nan

    std = x.std(ddof=1)

    if (
        not np.isfinite(std)
        or
        std == 0
    ):
        return np.nan

    return (
        x.mean()
        /
        (
            std
            /
            np.sqrt(n)
        )
    )


# ============================================================
# Factor bucket
# ============================================================

def factor_bucket(x):

    if not np.isfinite(x):
        return "NA"

    if x >= 0.6:
        return "F_HIGH"

    if x <= -0.6:
        return "F_LOW"

    return "F_MID"


# ============================================================
# Price bucket
# ============================================================

def price_bucket(x):

    if not np.isfinite(x):
        return "NA"

    if x >= 0.6:
        return "P_HIGH"

    if x <= -0.6:
        return "P_LOW"

    return "P_MID"


# ============================================================
# Build forward 5-day return from existing fwd1
#
# fwd1 at date t:
# return from close[t-1] -> close[t]
#
# fwd5 at t:
# compound t, t+1, ..., t+4
# ============================================================

def build_fwd5(g):

    g = g.sort_values(
        "trade_date"
    ).copy()

    r = g["fwd1"].values

    n = len(r)

    out = np.full(
        n,
        np.nan,
        dtype=np.float64
    )

    for i in range(n):

        if i + 5 > n:
            continue

        x = r[
            i:i + 5
        ]

        if not np.all(
            np.isfinite(x)
        ):
            continue

        out[i] = (
            np.prod(
                1.0 + x
            )
            -
            1.0
        )

    g["fwd5_calc"] = out

    return g


# ============================================================
# Add market-relative returns
#
# Cross-sectional industry average is removed every date.
#
# This is more appropriate because Factor22 is an
# industry-relative signal.
# ============================================================

def add_relative_returns(df):

    df = df.copy()

    df[
        "market_fwd1"
    ] = (
        df.groupby(
            "trade_date"
        )[
            "fwd1"
        ]
        .transform(
            "mean"
        )
    )

    df[
        "market_fwd5"
    ] = (
        df.groupby(
            "trade_date"
        )[
            "fwd5_calc"
        ]
        .transform(
            "mean"
        )
    )

    df[
        "rel_fwd1"
    ] = (
        df["fwd1"]
        -
        df["market_fwd1"]
    )

    df[
        "rel_fwd5"
    ] = (
        df["fwd5_calc"]
        -
        df["market_fwd5"]
    )

    return df


# ============================================================
# Period
# ============================================================

def add_period(df):

    df = df.copy()

    df["year"] = (
        df["trade_date"]
        .astype(str)
        .str[:4]
        .astype(int)
    )

    df["period"] = np.where(
        df["year"] <= 2020,
        "2016-2020",
        "2021-2024H1"
    )

    return df


# ============================================================
# Side × state analysis
# ============================================================

def side_state_summary(
    df,
    state_col,
    state_name,
    period_name="ALL"
):

    x = df.copy()

    if period_name != "ALL":

        x = x[
            x["period"]
            ==
            period_name
        ].copy()

    x[
        "factor_bucket"
    ] = (
        x["factor22"]
        .apply(
            factor_bucket
        )
    )

    x[
        "price_bucket"
    ] = (
        x[state_col]
        .apply(
            price_bucket
        )
    )

    x = x[
        x[
            "factor_bucket"
        ].isin(
            [
                "F_HIGH",
                "F_LOW",
            ]
        )
    ]

    rows = []

    for (
        fb,
        pb
    ), g in x.groupby(
        [
            "factor_bucket",
            "price_bucket",
        ]
    ):

        rows.append(
            {
                "period":
                    period_name,

                "state":
                    state_name,

                "factor_side":
                    fb,

                "price_state":
                    pb,

                "n":
                    len(g),

                "rel1_mean":
                    g[
                        "rel_fwd1"
                    ].mean(),

                "rel1_bp":
                    10000.0
                    *
                    g[
                        "rel_fwd1"
                    ].mean(),

                "rel1_t":
                    t_stat(
                        g[
                            "rel_fwd1"
                        ]
                    ),

                "rel5_mean":
                    g[
                        "rel_fwd5"
                    ].mean(),

                "rel5_bp":
                    10000.0
                    *
                    g[
                        "rel_fwd5"
                    ].mean(),

                "rel5_t":
                    t_stat(
                        g[
                            "rel_fwd5"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Daily long-short spread
#
# Long = F_LOW
# Short = F_HIGH
#
# spread = long - short
# ============================================================

def daily_spread_summary(
    df,
    state_col,
    state_name,
    period_name="ALL"
):

    x = df.copy()

    if period_name != "ALL":

        x = x[
            x["period"]
            ==
            period_name
        ].copy()

    x[
        "factor_bucket"
    ] = (
        x["factor22"]
        .apply(
            factor_bucket
        )
    )

    x[
        "price_bucket"
    ] = (
        x[state_col]
        .apply(
            price_bucket
        )
    )

    rows = []

    for pb in [
        "P_LOW",
        "P_MID",
        "P_HIGH",
    ]:

        z = x[
            x["price_bucket"]
            ==
            pb
        ].copy()

        daily = (
            z.groupby(
                [
                    "trade_date",
                    "factor_bucket",
                ]
            )[
                [
                    "rel_fwd1",
                    "rel_fwd5",
                ]
            ]
            .mean()
            .reset_index()
        )

        p1 = daily.pivot(
            index="trade_date",
            columns="factor_bucket",
            values="rel_fwd1"
        )

        p5 = daily.pivot(
            index="trade_date",
            columns="factor_bucket",
            values="rel_fwd5"
        )

        if (
            "F_HIGH" not in p1.columns
            or
            "F_LOW" not in p1.columns
        ):
            continue

        s1 = (
            p1["F_LOW"]
            -
            p1["F_HIGH"]
        ).dropna()

        s5 = (
            p5["F_LOW"]
            -
            p5["F_HIGH"]
        ).dropna()

        rows.append(
            {
                "period":
                    period_name,

                "state":
                    state_name,

                "price_state":
                    pb,

                "n_days_1":
                    len(s1),

                "spread1_bp":
                    s1.mean()
                    *
                    10000.0,

                "spread1_t":
                    t_stat(s1),

                "n_days_5":
                    len(s5),

                "spread5_bp":
                    s5.mean()
                    *
                    10000.0,

                "spread5_t":
                    t_stat(s5),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Pure Factor22 side decomposition
# ============================================================

def factor_side_summary(
    df,
    period_name
):

    x = df.copy()

    if period_name != "ALL":

        x = x[
            x["period"]
            ==
            period_name
        ]

    x = x.copy()

    x[
        "factor_bucket"
    ] = (
        x["factor22"]
        .apply(
            factor_bucket
        )
    )

    x = x[
        x[
            "factor_bucket"
        ].isin(
            [
                "F_HIGH",
                "F_LOW",
            ]
        )
    ]

    rows = []

    for fb, g in x.groupby(
        "factor_bucket"
    ):

        rows.append(
            {
                "period":
                    period_name,

                "side":
                    fb,

                "n":
                    len(g),

                "rel1_bp":
                    g[
                        "rel_fwd1"
                    ].mean()
                    *
                    10000.0,

                "rel1_t":
                    t_stat(
                        g[
                            "rel_fwd1"
                        ]
                    ),

                "rel5_bp":
                    g[
                        "rel_fwd5"
                    ].mean()
                    *
                    10000.0,

                "rel5_t":
                    t_stat(
                        g[
                            "rel_fwd5"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    df = pd.read_csv(
        INPUT
    )

    df = (
        df.groupby(
            "industry",
            group_keys=False
        )
        .apply(
            build_fwd5
        )
        .reset_index(
            drop=True
        )
    )

    df = add_relative_returns(
        df
    )

    df = add_period(
        df
    )

    print("")
    print(
        "rows:",
        len(df)
    )

    print(
        "dates:",
        df[
            "trade_date"
        ].nunique()
    )

    print(
        "fwd5 valid:",
        df[
            "fwd5_calc"
        ].notna().sum()
    )


    states = [
        (
            "ret5_rank",
            "RET5"
        ),

        (
            "accel5_rank",
            "ACCEL5"
        ),

        (
            "overnight5_rank",
            "OVERNIGHT5"
        ),

        (
            "intraday5_rank",
            "INTRADAY5"
        ),
    ]


    periods = [
        "ALL",
        "2016-2020",
        "2021-2024H1",
    ]


    # ========================================================
    # Factor22 side decomposition
    # ========================================================

    side_results = []

    for period in periods:

        side_results.append(
            factor_side_summary(
                df,
                period
            )
        )

    side_results = pd.concat(
        side_results,
        ignore_index=True
    )

    print("")
    print("=" * 90)
    print("FACTOR22 SIDE DECOMPOSITION")
    print("returns are industry-relative")
    print("=" * 90)

    print(
        side_results.to_string(
            index=False
        )
    )

    side_results.to_csv(
        "outputs/"
        "factor22_side_summary.csv",
        index=False
    )


    # ========================================================
    # Side × price state
    # ========================================================

    detail_results = []

    for period in periods:

        for state_col, state_name in states:

            detail_results.append(
                side_state_summary(
                    df,
                    state_col,
                    state_name,
                    period
                )
            )

    detail_results = pd.concat(
        detail_results,
        ignore_index=True
    )

    print("")
    print("=" * 90)
    print("FACTOR22 SIDE × PRICE STATE")
    print("=" * 90)

    print(
        detail_results.to_string(
            index=False
        )
    )

    detail_results.to_csv(
        "outputs/"
        "factor22_side_state_v2.csv",
        index=False
    )


    # ========================================================
    # Conditional long-short
    # ========================================================

    spread_results = []

    for period in periods:

        for state_col, state_name in states:

            spread_results.append(
                daily_spread_summary(
                    df,
                    state_col,
                    state_name,
                    period
                )
            )

    spread_results = pd.concat(
        spread_results,
        ignore_index=True
    )

    print("")
    print("=" * 90)
    print("CONDITIONAL LONG-SHORT")
    print("=" * 90)

    print(
        spread_results.to_string(
            index=False
        )
    )

    spread_results.to_csv(
        "outputs/"
        "factor22_conditional_spread_v2.csv",
        index=False
    )


    # ========================================================
    # Save enriched raw data
    # ========================================================

    df.to_csv(
        "outputs/"
        "factor22_bucket_enriched.csv",
        index=False
    )


if __name__ == "__main__":
    main()