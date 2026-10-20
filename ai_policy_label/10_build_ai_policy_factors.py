# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ============================================================
# Cross-sectional rank
# ============================================================

def cross_sectional_rank(series):
    """
    每个交易日/自然日内部，对行业做横截面 rank。

    返回约 [-1, +1]：
        最低 -> 接近 -1
        中位 -> 接近 0
        最高 -> 接近 +1

    NaN 保持 NaN。
    """

    valid = series.notna()

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=np.float64,
    )

    n = int(valid.sum())

    if n <= 1:
        return result

    ranked = (
        series.loc[valid]
        .rank(
            method="average",
            pct=True,
        )
    )

    result.loc[valid] = (
        ranked * 2.0 - 1.0
    )

    return result


# ============================================================
# Full calendar panel
# ============================================================

def build_full_calendar_panel(df):
    """
    构造：
        每个自然日 × 每个行业

    没有人民日报相关新闻的行业当天 attention = 0。

    这一点很重要：
    我们做的是自然日信号，
    周末也必须保留。
    """

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["industry"] = (
        df["industry"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(6)
    )

    dates = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="D",
    )

    industries = sorted(
        df["industry"].dropna().unique()
    )

    full_index = pd.MultiIndex.from_product(
        [
            dates,
            industries,
        ],
        names=[
            "date",
            "industry",
        ],
    )

    value_cols = [
        "positive_attention",
        "negative_attention",
        "net_attention",
    ]

    panel = (
        df[
            [
                "date",
                "industry",
            ]
            + value_cols
        ]
        .groupby(
            [
                "date",
                "industry",
            ],
            as_index=False,
        )[value_cols]
        .sum()
        .set_index(
            [
                "date",
                "industry",
            ]
        )
        .reindex(full_index)
    )

    # 没有文章 -> attention 为 0
    panel[value_cols] = (
        panel[value_cols]
        .fillna(0.0)
    )

    return panel.reset_index()


# ============================================================
# Rolling factors
# ============================================================

def add_industry_rolling_factors(
    g,
    short_window,
    history_window,
    min_history,
):
    """
    单个行业的时间序列。

    surprise 的历史均值/std 必须 shift(1)，
    避免把当天自己放进 benchmark。
    """

    g = (
        g
        .sort_values("date")
        .copy()
    )

    # ------------------------------------------
    # 20 natural-day accumulated attention
    # ------------------------------------------

    g["positive_20"] = (
        g["positive_attention"]
        .rolling(
            short_window,
            min_periods=1,
        )
        .sum()
    )

    g["negative_20"] = (
        g["negative_attention"]
        .rolling(
            short_window,
            min_periods=1,
        )
        .sum()
    )

    g["net_20"] = (
        g["net_attention"]
        .rolling(
            short_window,
            min_periods=1,
        )
        .sum()
    )

    # ------------------------------------------
    # Surprise helper
    # ------------------------------------------

    def calc_surprise(x):

        historical = x.shift(1)

        mean = (
            historical
            .rolling(
                history_window,
                min_periods=min_history,
            )
            .mean()
        )

        std = (
            historical
            .rolling(
                history_window,
                min_periods=min_history,
            )
            .std(ddof=1)
        )

        std = std.replace(
            0.0,
            np.nan,
        )

        return (
            (x - mean)
            / std
        )

    g[
        "positive_surprise_20"
    ] = calc_surprise(
        g["positive_20"]
    )

    g[
        "negative_surprise_20"
    ] = calc_surprise(
        g["negative_20"]
    )

    g[
        "net_surprise_20"
    ] = calc_surprise(
        g["net_20"]
    )

    return g


# ============================================================
# Main factor construction
# ============================================================

def build_factors(
    df,
    short_window,
    history_window,
    min_history,
):

    print(
        "[1] building full calendar panel..."
    )

    panel = (
        build_full_calendar_panel(
            df
        )
    )

    print(
        "calendar rows:",
        len(panel)
    )

    print(
        "date range:",
        panel["date"].min(),
        "->",
        panel["date"].max()
    )

    print(
        "industries:",
        panel["industry"].nunique()
    )

    print(
        "[2] calculating rolling factors..."
    )

    pieces = []

    for industry, g in panel.groupby(
        "industry",
        sort=False,
    ):

        x = (
            add_industry_rolling_factors(
                g,
                short_window,
                history_window,
                min_history,
            )
        )

        pieces.append(x)

    factor = pd.concat(
        pieces,
        ignore_index=True,
    )

    factor = (
        factor
        .sort_values(
            [
                "date",
                "industry",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "[3] calculating cross-sectional ranks..."
    )

    rank_sources = {
        "positive_attention_20_rank":
            "positive_20",

        "negative_attention_20_rank":
            "negative_20",

        "net_attention_20_rank":
            "net_20",

        "positive_surprise_20_rank":
            "positive_surprise_20",

        "negative_surprise_20_rank":
            "negative_surprise_20",

        "net_surprise_20_rank":
            "net_surprise_20",
    }

    for output_col, source_col in (
        rank_sources.items()
    ):

        factor[output_col] = (
            factor
            .groupby(
                "date",
                group_keys=False,
            )[source_col]
            .apply(
                cross_sectional_rank
            )
        )

    # YYYYMMDD integer
    factor["date_int"] = (
        factor["date"]
        .dt.strftime("%Y%m%d")
        .astype(np.int64)
    )

    return factor


# ============================================================
# Diagnostics
# ============================================================

def print_diagnostics(df):

    print("")
    print("=" * 70)
    print("AI POLICY FACTOR DIAGNOSTICS")
    print("=" * 70)

    print(
        "rows:",
        len(df)
    )

    print(
        "dates:",
        df["date"].nunique()
    )

    print(
        "industries:",
        df["industry"].nunique()
    )

    print(
        "date:",
        df["date"].min(),
        "->",
        df["date"].max()
    )

    cols = [
        "positive_attention",
        "negative_attention",
        "net_attention",
        "positive_20",
        "negative_20",
        "net_20",
        "positive_surprise_20",
        "negative_surprise_20",
        "net_surprise_20",
        "positive_surprise_20_rank",
        "negative_surprise_20_rank",
        "net_surprise_20_rank",
        "net_attention_20_rank",
    ]

    print("")
    print(
        df[cols]
        .describe()
        .T[
            [
                "count",
                "mean",
                "std",
                "min",
                "50%",
                "max",
            ]
        ]
    )

    print("")
    print(
        "latest sample:"
    )

    latest_date = (
        df["date"].max()
    )

    print(
        df.loc[
            df["date"] == latest_date,
            [
                "date",
                "industry",
                "net_20",
                "net_surprise_20",
                "net_surprise_20_rank",
            ]
        ]
        .sort_values(
            "net_surprise_20_rank",
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )

    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--input",
        default=None,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:

        cfg = yaml.safe_load(f)

    ai_cfg = cfg.get(
        "ai_factor",
        {}
    )

    input_path = (
        args.input
        or ai_cfg.get(
            "input_daily"
        )
        or cfg["output"][
            "daily_parquet"
        ]
    )

    output_path = (
        args.output
        or ai_cfg.get(
            "output_parquet",
            "../data/factor/"
            "people_daily_ai_policy_factor.parquet",
        )
    )

    short_window = int(
        ai_cfg.get(
            "short_window",
            20,
        )
    )

    history_window = int(
        ai_cfg.get(
            "history_window",
            365,
        )
    )

    min_history = int(
        ai_cfg.get(
            "min_history",
            120,
        )
    )

    print(
        "input:",
        input_path
    )

    print(
        "output:",
        output_path
    )

    print(
        "short_window:",
        short_window
    )

    print(
        "history_window:",
        history_window
    )

    print(
        "min_history:",
        min_history
    )

    df = pd.read_parquet(
        input_path
    )

    required = {
        "date",
        "industry",
        "positive_attention",
        "negative_attention",
        "net_attention",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise KeyError(
            "missing columns: %s"
            % sorted(missing)
        )

    factor = build_factors(
        df,
        short_window,
        history_window,
        min_history,
    )

    print_diagnostics(
        factor
    )

    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    factor.to_parquet(
        output_path,
        index=False,
    )

    print("")
    print(
        "saved:",
        output_path
    )


if __name__ == "__main__":
    main()