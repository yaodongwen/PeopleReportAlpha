# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


INPUT = (
    "outputs/"
    "factor22_bucket_raw.csv"
)

OUTPUT_DETAIL = (
    "outputs/"
    "factor22_bucket_summary.csv"
)


# ============================================================
# t-stat
# ============================================================

def t_stat(x):

    x = pd.Series(
        x
    ).dropna()

    n = len(x)

    if n < 2:
        return np.nan

    std = x.std(
        ddof=1
    )

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
# Factor22 bucket
#
# High:
#     > 0.6
#
# Low:
#     < -0.6
#
# Middle ignored in core study.
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
#
# price rank:
# top/bottom 20% roughly correspond to +/-0.6
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
# One variable summary
# ============================================================

def summarize_one(
    df,
    price_col,
    label
):

    tmp = df.copy()

    tmp[
        "factor_bucket"
    ] = (
        tmp["factor22"]
        .apply(
            factor_bucket
        )
    )

    tmp[
        "price_bucket"
    ] = (
        tmp[
            price_col
        ]
        .apply(
            price_bucket
        )
    )

    # core extreme×extreme table
    tmp = tmp[
        tmp[
            "factor_bucket"
        ].isin(
            [
                "F_HIGH",
                "F_LOW",
            ]
        )
        &
        tmp[
            "price_bucket"
        ].isin(
            [
                "P_HIGH",
                "P_LOW",
            ]
        )
    ].copy()

    rows = []

    for (
        fb,
        pb
    ), g in tmp.groupby(
        [
            "factor_bucket",
            "price_bucket",
        ]
    ):

        row = {
            "state":
                label,

            "factor_bucket":
                fb,

            "price_bucket":
                pb,

            "n":
                len(g),

            "fwd1_mean":
                g[
                    "fwd1"
                ].mean(),

            "fwd1_t":
                t_stat(
                    g[
                        "fwd1"
                    ]
                ),

            "fwd5_mean":
                g[
                    "fwd5"
                ].mean(),

            "fwd5_t":
                t_stat(
                    g[
                        "fwd5"
                    ]
                ),

            "factor22_mean":
                g[
                    "factor22"
                ].mean(),

            "price_rank_mean":
                g[
                    price_col
                ].mean(),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Long-short oriented spread
#
# Because current final sign = -1:
#
# F_HIGH:
#     should have lower future return
#
# F_LOW:
#     should have higher future return
#
# strategy spread:
#
# F_LOW - F_HIGH
# ============================================================

def strategy_spread(
    df,
    price_col,
    label
):

    tmp = df.copy()

    tmp[
        "factor_bucket"
    ] = (
        tmp[
            "factor22"
        ]
        .apply(
            factor_bucket
        )
    )

    tmp[
        "price_bucket"
    ] = (
        tmp[
            price_col
        ]
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

        x = tmp[
            tmp[
                "price_bucket"
            ]
            ==
            pb
        ].copy()

        # ----------------------------------------------------
        # Daily cross-sectional average return by F22 side
        # ----------------------------------------------------

        grouped = (
            x.groupby(
                [
                    "trade_date",
                    "factor_bucket",
                ]
            )[
                [
                    "fwd1",
                    "fwd5",
                ]
            ]
            .mean()
            .reset_index()
        )

        f1 = grouped.pivot(
            index="trade_date",
            columns="factor_bucket",
            values="fwd1",
        )

        f5 = grouped.pivot(
            index="trade_date",
            columns="factor_bucket",
            values="fwd5",
        )

        if (
            "F_LOW" not in f1.columns
            or
            "F_HIGH" not in f1.columns
        ):
            continue

        spread1 = (
            f1[
                "F_LOW"
            ]
            -
            f1[
                "F_HIGH"
            ]
        ).dropna()

        spread5 = (
            f5[
                "F_LOW"
            ]
            -
            f5[
                "F_HIGH"
            ]
        ).dropna()

        rows.append(
            {
                "state":
                    label,

                "price_bucket":
                    pb,

                "n_days":
                    len(spread1),

                "spread1_mean":
                    spread1.mean(),

                "spread1_t":
                    t_stat(
                        spread1
                    ),

                "spread5_mean":
                    spread5.mean(),

                "spread5_t":
                    t_stat(
                        spread5
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
        "industries:",
        df[
            "industry"
        ].nunique()
    )

    print(
        "date range:",
        df[
            "trade_date"
        ].min(),
        "->",
        df[
            "trade_date"
        ].max()
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


    # ========================================================
    # Extreme x extreme
    # ========================================================

    detail_list = []

    for col, label in states:

        result = summarize_one(
            df,
            col,
            label
        )

        detail_list.append(
            result
        )


    detail = pd.concat(
        detail_list,
        ignore_index=True
    )

    detail.to_csv(
        OUTPUT_DETAIL,
        index=False
    )


    print("")
    print("=" * 90)
    print("EXTREME FACTOR22 × PRICE STATE")
    print("=" * 90)

    print(
        detail.to_string(
            index=False
        )
    )


    # ========================================================
    # Strategy-oriented spreads
    # ========================================================

    spread_list = []

    for col, label in states:

        result = strategy_spread(
            df,
            col,
            label
        )

        spread_list.append(
            result
        )


    spread = pd.concat(
        spread_list,
        ignore_index=True
    )


    print("")
    print("=" * 90)
    print("FACTOR22 LONG-SHORT SPREAD CONDITIONAL ON PRICE STATE")
    print("")
    print("spread = F_LOW future return - F_HIGH future return")
    print("positive = consistent with sign=-1")
    print("=" * 90)

    print(
        spread.to_string(
            index=False
        )
    )


    spread.to_csv(
        "outputs/"
        "factor22_bucket_spread.csv",
        index=False
    )


    # ========================================================
    # Basic Factor22 alone
    # ========================================================

    print("")
    print("=" * 90)
    print("FACTOR22 ALONE")
    print("=" * 90)

    tmp = df.copy()

    tmp[
        "factor_bucket"
    ] = (
        tmp[
            "factor22"
        ]
        .apply(
            factor_bucket
        )
    )

    base = (
        tmp[
            tmp[
                "factor_bucket"
            ].isin(
                [
                    "F_HIGH",
                    "F_LOW",
                ]
            )
        ]
        .groupby(
            "factor_bucket"
        )
        .agg(
            n=(
                "fwd1",
                "count"
            ),
            fwd1_mean=(
                "fwd1",
                "mean"
            ),
            fwd5_mean=(
                "fwd5",
                "mean"
            ),
        )
    )

    print(
        base
    )


if __name__ == "__main__":
    main()