# -*- coding: utf-8 -*-

"""
Step 5

industry_daily.parquet
->
构造行业政策关注因子

输出：
industry_policy_factor.parquet
"""

import numpy as np
import pandas as pd

from config import FACTOR_DIR


INPUT_FILE = (
    FACTOR_DIR
    / "industry_daily.parquet"
)

OUTPUT_FILE = (
    FACTOR_DIR
    / "industry_policy_factor.parquet"
)


# ============================================================
# 参数
# ============================================================

SHORT_WINDOW = 20
LONG_WINDOW = 60

# surprise 的历史基准
HISTORY_WINDOW = 365

# 至少有这么多历史数据才计算 surprise
MIN_HISTORY = 120


def load_daily_signal():
    """读取行业日度关注度。"""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    df = pd.read_parquet(
        INPUT_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def make_complete_panel(df):
    """
    补齐：
        每一天 × 每一个行业

    某行业某天没有匹配文章：
        attention = 0

    这是非常重要的。
    """

    dates = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="D"
    )

    industries = sorted(
        df["industry"].unique()
    )

    index = pd.MultiIndex.from_product(
        [
            dates,
            industries,
        ],
        names=[
            "date",
            "industry",
        ]
    )

    panel = (
        df[
            [
                "date",
                "industry",
                "framework_code",
                "index_code",
                "attention",
            ]
        ]
        .set_index(
            [
                "date",
                "industry",
            ]
        )
        .reindex(index)
        .reset_index()
    )

    # 没有行业新闻 = 0
    panel["attention"] = (
        panel["attention"]
        .fillna(0.0)
    )

    # 行业代码向前后填充
    panel["framework_code"] = (
        panel
        .groupby("industry")[
            "framework_code"
        ]
        .transform(
            lambda x: (
                x.ffill()
                .bfill()
            )
        )
    )

    panel["index_code"] = (
        panel
        .groupby("industry")[
            "index_code"
        ]
        .transform(
            lambda x: (
                x.ffill()
                .bfill()
            )
        )
    )

    return panel


def rolling_sum(
    series,
    window
):
    """过去 N 日关注度总和。"""

    return (
        series
        .rolling(
            window=window,
            min_periods=1
        )
        .sum()
    )


def build_attention_factors(group):
    """单个行业的时间序列因子。"""

    x = (
        group
        .sort_values("date")
        .copy()
    )

    # --------------------------------------------
    # 20日 / 60日累计关注
    # --------------------------------------------

    x["attention_20"] = rolling_sum(
        x["attention"],
        SHORT_WINDOW
    )

    x["attention_60"] = rolling_sum(
        x["attention"],
        LONG_WINDOW
    )

    # --------------------------------------------
    # 关注变化：
    #
    # 最近20日
    # -
    # 前20日
    # --------------------------------------------

    previous_20 = (
        x["attention_20"]
        .shift(
            SHORT_WINDOW
        )
    )

    x["delta_attention"] = (
        x["attention_20"]
        -
        previous_20
    )

    # --------------------------------------------
    # Surprise
    #
    # 当前20日关注度
    # 相对过去365日自己的历史分布
    #
    # 注意：
    # shift(1) 保证基准历史里
    # 不含当天当前值
    # --------------------------------------------

    history = (
        x["attention_20"]
        .shift(1)
    )

    history_mean = (
        history
        .rolling(
            HISTORY_WINDOW,
            min_periods=MIN_HISTORY
        )
        .mean()
    )

    history_std = (
        history
        .rolling(
            HISTORY_WINDOW,
            min_periods=MIN_HISTORY
        )
        .std()
    )

    x["surprise_20"] = (
        x["attention_20"]
        -
        history_mean
    ) / history_std

    x.loc[
        history_std <= 1e-12,
        "surprise_20"
    ] = np.nan

    return x


def build_factors(panel):
    """对所有行业构造因子。"""

    parts = []

    for industry, group in panel.groupby(
        "industry",
        sort=False
    ):

        result = build_attention_factors(
            group
        )

        parts.append(
            result
        )

    return pd.concat(
        parts,
        ignore_index=True
    )


def print_summary(df):

    print()
    print("=" * 80)
    print("POLICY FACTOR SUMMARY")
    print("=" * 80)

    print(
        "rows:",
        f"{len(df):,}"
    )

    print(
        "date:",
        df["date"].min(),
        "->",
        df["date"].max()
    )

    print(
        "industries:",
        df["industry"].nunique()
    )

    print()

    cols = [
        "attention",
        "attention_20",
        "attention_60",
        "delta_attention",
        "surprise_20",
    ]

    print(
        df[cols]
        .describe()
        .round(6)
        .to_string()
    )


def main():

    daily = load_daily_signal()

    print(
        "[input]",
        f"{len(daily):,}"
    )

    panel = make_complete_panel(
        daily
    )

    print(
        "[complete panel]",
        f"{len(panel):,}"
    )

    factors = build_factors(
        panel
    )

    factors = (
        factors
        .sort_values(
            [
                "date",
                "industry",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    factors.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print_summary(
        factors
    )

    print()
    print(
        "[done]",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()