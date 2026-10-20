# -*- coding: utf-8 -*-

"""
Step 6

把行业政策因子整理成 AlphaSim 友好的格式。

输出：
    people_daily_industry_factor.csv
    people_daily_industry_factor.parquet

同时生成：
    原始因子
    行业横截面 rank 因子
"""

import numpy as np
import pandas as pd

from config import FACTOR_DIR


INPUT_FILE = (
    FACTOR_DIR
    / "industry_policy_factor.parquet"
)

CSV_FILE = (
    FACTOR_DIR
    / "people_daily_industry_factor.csv"
)

PARQUET_FILE = (
    FACTOR_DIR
    / "people_daily_industry_factor_alphasim.parquet"
)


FACTOR_COLUMNS = [
    "attention_20",
    "attention_60",
    "delta_attention",
    "surprise_20",
]


def cross_sectional_rank(series):
    """
    每天在行业之间做 rank。

    输出大致范围：
        -1 ~ +1

    小值 -> -1
    大值 -> +1
    """

    valid = series.notna()

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    n = valid.sum()

    if n <= 1:
        return result

    rank = (
        series[valid]
        .rank(
            method="average"
        )
    )

    result.loc[valid] = (
        2.0
        * (rank - 1.0)
        / (n - 1.0)
        - 1.0
    )

    return result


def add_rank_factors(df):

    df = df.copy()

    for col in FACTOR_COLUMNS:

        rank_col = (
            col
            + "_rank"
        )

        df[rank_col] = (
            df
            .groupby("date")[col]
            .transform(
                cross_sectional_rank
            )
        )

    return df


def clean_codes(df):

    df = df.copy()

    df["framework_code"] = (
        df["framework_code"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False
        )
        .str.zfill(6)
    )

    df["index_code"] = (
        df["index_code"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False
        )
        .str.zfill(6)
    )

    return df


def main():

    df = pd.read_parquet(
        INPUT_FILE
    )

    df["date"] = (
        pd.to_datetime(
            df["date"]
        )
        .dt.strftime(
            "%Y%m%d"
        )
        .astype(int)
    )

    df = clean_codes(
        df
    )

    df = add_rank_factors(
        df
    )

    keep_cols = [
        "date",
        "industry",
        "framework_code",
        "index_code",
        "attention_20",
        "attention_60",
        "delta_attention",
        "surprise_20",
        "attention_20_rank",
        "attention_60_rank",
        "delta_attention_rank",
        "surprise_20_rank",
    ]

    df = (
        df[keep_cols]
        .sort_values(
            [
                "date",
                "framework_code",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df.to_csv(
        CSV_FILE,
        index=False
    )

    df.to_parquet(
        PARQUET_FILE,
        index=False
    )

    print(
        "[rows]",
        f"{len(df):,}"
    )

    print(
        "[date]",
        df["date"].min(),
        "->",
        df["date"].max()
    )

    print(
        "[industries]",
        df["industry"].nunique()
    )

    print()

    print(
        df.head(20)
        .to_string(index=False)
    )

    print()
    print(
        "[done]",
        CSV_FILE
    )

    print(
        "[done]",
        PARQUET_FILE
    )


if __name__ == "__main__":
    main()