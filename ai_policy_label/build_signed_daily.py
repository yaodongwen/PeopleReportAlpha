# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def find_date_column(df, preferred="date"):
    """
    兼容 merge 后的：
        date
        date_y
        date_x
    """

    candidates = [
        preferred,
        "date_y",
        "date_x",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        "Cannot find date column. Available columns: %s"
        % df.columns.tolist()
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config.yaml"
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8"
    ) as f:
        cfg = yaml.safe_load(f)

    inp = cfg["input"]
    out = cfg["output"]

    input_path = out["final_parquet"]
    output_path = out["daily_parquet"]

    join_key = inp["join_key"]
    industry_name_col = inp["industry_col"]

    industry_code_col = "framework_code"

    article_weight_col = inp.get(
        "article_weight_col",
        "article_weight"
    )

    configured_date_col = inp.get(
        "date_col",
        "date"
    )

    print(
        "[1] loading:",
        input_path
    )

    df = pd.read_parquet(
        input_path
    )

    print(
        "rows:",
        len(df)
    )

    print(
        "columns:",
        df.columns.tolist()
    )

    # ========================================================
    # 自动寻找日期列
    # ========================================================

    date_col = find_date_column(
        df,
        configured_date_col
    )

    print(
        "using date column:",
        date_col
    )

    df["date"] = (
        pd.to_datetime(
            df[date_col],
            errors="coerce"
        )
        .dt.normalize()
    )

    invalid_date = int(
        df["date"].isna().sum()
    )

    if invalid_date > 0:
        print(
            "warning: invalid dates:",
            invalid_date
        )

    # ========================================================
    # 检查必要字段
    # ========================================================

    required = [
        join_key,
        industry_code_col,
        "effective_direction",
        "confidence",
        "relevance",
        "policy_strength",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            "missing columns: %s"
            % missing
        )

    # article_weight 如果不存在，就默认1
    if article_weight_col not in df.columns:

        print(
            "warning:",
            article_weight_col,
            "not found, use 1.0"
        )

        df[article_weight_col] = 1.0

    # ========================================================
    # 清理数值
    # ========================================================

    numeric_cols = [
        article_weight_col,
        "effective_direction",
        "confidence",
        "relevance",
        "policy_strength",
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df[article_weight_col] = (
        df[article_weight_col]
        .fillna(1.0)
    )

    df["confidence"] = (
        df["confidence"]
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    df["relevance"] = (
        df["relevance"]
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    df["policy_strength"] = (
        df["policy_strength"]
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    df["effective_direction"] = (
        df["effective_direction"]
        .fillna(0)
        .astype(int)
    )

    # ========================================================
    # AI 权重
    # ========================================================

    agg_cfg = cfg.get(
        "aggregation",
        {}
    )

    df["ai_weight"] = (
        df[article_weight_col]
        .astype(float)
    )

    if agg_cfg.get(
        "use_confidence",
        True
    ):
        df["ai_weight"] *= (
            df["confidence"]
        )

    if agg_cfg.get(
        "use_relevance",
        True
    ):
        df["ai_weight"] *= (
            df["relevance"]
        )

    if agg_cfg.get(
        "use_policy_strength",
        True
    ):
        df["ai_weight"] *= (
            df["policy_strength"]
        )

    # ========================================================
    # 正 / 负 / 净方向
    # ========================================================

    df["positive_weight"] = np.where(
        df["effective_direction"] == 1,
        df["ai_weight"],
        0.0,
    )

    # negative_attention 表示“负面强度”
    # 所以这里保存正数 magnitude
    df["negative_weight"] = np.where(
        df["effective_direction"] == -1,
        df["ai_weight"],
        0.0,
    )

    # net：
    # 正面为正
    # 负面为负
    df["signed_weight"] = (
        df["ai_weight"]
        * df["effective_direction"]
    )

    # ========================================================
    # denominator：每天人民日报全部文章数量
    #
    # 注意：
    # 不应该只统计有行业标签的文章。
    # 所以从 news_clean.parquet 单独计算。
    # ========================================================

    news_path = inp["news_path"]

    print(
        "[2] loading news denominator:",
        news_path
    )

    news = pd.read_parquet(
        news_path
    )

    news_date_col = (
        configured_date_col
        if configured_date_col
        in news.columns
        else find_date_column(
            news,
            configured_date_col
        )
    )

    news["date"] = (
        pd.to_datetime(
            news[news_date_col],
            errors="coerce"
        )
        .dt.normalize()
    )

    if join_key in news.columns:

        total_articles = (
            news
            .dropna(
                subset=["date"]
            )
            .groupby("date")[
                join_key
            ]
            .nunique()
            .rename(
                "total_articles"
            )
            .reset_index()
        )

    else:

        total_articles = (
            news
            .dropna(
                subset=["date"]
            )
            .groupby("date")
            .size()
            .rename(
                "total_articles"
            )
            .reset_index()
        )

    # ========================================================
    # 日期 × 行业聚合
    # ========================================================

    print(
        "[3] aggregating date x industry..."
    )

    valid = df[
        df["date"].notna()
        & df[industry_name_col].notna()
    ].copy()

    daily = (
        valid
        .groupby(
            [
                "date",
                industry_code_col,
            ],
            as_index=False,
        )
        .agg(
            positive_weight=(
                "positive_weight",
                "sum"
            ),

            negative_weight=(
                "negative_weight",
                "sum"
            ),

            signed_weight=(
                "signed_weight",
                "sum"
            ),

            labeled_articles=(
                join_key,
                "nunique"
            ),
        )
    )

    daily = daily.merge(
        total_articles,
        on="date",
        how="left",
        validate="many_to_one",
    )

    # 避免除0
    denominator = (
        daily["total_articles"]
        .replace(
            0,
            np.nan
        )
    )

    daily[
        "positive_attention"
    ] = (
        daily["positive_weight"]
        / denominator
    )

    daily[
        "negative_attention"
    ] = (
        daily["negative_weight"]
        / denominator
    )

    daily[
        "net_attention"
    ] = (
        daily["signed_weight"]
        / denominator
    )

    # 统一行业编码格式
    daily["industry"] = (
        pd.to_numeric(
            daily[industry_code_col],
            errors="coerce"
        )
        .astype("Int64")
        .astype(str)
        .str.zfill(6)
    )

    if industry_code_col != "industry":
        daily = daily.drop(
            columns=[industry_code_col]
        )

    daily = (
        daily
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

    # ========================================================
    # diagnostics
    # ========================================================

    print("")
    print("=" * 70)
    print("SIGNED DAILY SUMMARY")
    print("=" * 70)

    print(
        "rows:",
        len(daily)
    )

    print(
        "dates:",
        daily[
            "date"
        ].nunique()
    )

    print(
        "date range:",
        daily[
            "date"
        ].min(),
        "->",
        daily[
            "date"
        ].max()
    )

    print(
        "industries:",
        daily[
            "industry"
        ].nunique()
    )

    print("")

    print(
        daily[
            [
                "positive_attention",
                "negative_attention",
                "net_attention",
            ]
        ]
        .describe()
        .T
    )

    print("")

    print(
        "direction counts:"
    )

    print(
        df[
            "effective_direction"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print("=" * 70)

    # ========================================================
    # 保存
    # ========================================================

    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    daily.to_parquet(
        output_path,
        index=False
    )

    print("")
    print(
        "saved:",
        output_path
    )


if __name__ == "__main__":
    main()