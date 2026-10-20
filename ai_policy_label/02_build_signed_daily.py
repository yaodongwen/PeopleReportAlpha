# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def find_date_column(df, preferred="date"):

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

    ai_cfg = cfg.get(
        "ai_factor",
        {}
    )

    input_path = out["final_parquet"]
    output_path = out["daily_parquet"]

    join_key = inp["join_key"]

    # ------------------------------------------------------
    # AI分类时使用中文industry，
    # 因子层统一使用framework_code
    # ------------------------------------------------------

    industry_code_col = "framework_code"

    article_weight_col = inp.get(
        "article_weight_col",
        "article_weight"
    )

    configured_date_col = inp.get(
        "date_col",
        "date"
    )

    # ------------------------------------------------------
    # 高置信负面阈值
    # ------------------------------------------------------

    high_confidence = float(
        ai_cfg.get(
            "high_negative_confidence",
            0.80
        )
    )

    high_relevance = float(
        ai_cfg.get(
            "high_negative_relevance",
            0.70
        )
    )

    high_strength = float(
        ai_cfg.get(
            "high_negative_policy_strength",
            0.60
        )
    )

    constraint_types = set(
        ai_cfg.get(
            "constraint_policy_types",
            [
                "regulation",
                "restriction",
                "risk",
            ]
        )
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

    # ======================================================
    # 必要字段
    # ======================================================

    required = [
        join_key,
        industry_code_col,
        "effective_direction",
        "confidence",
        "relevance",
        "policy_strength",
        "policy_type",
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

    if article_weight_col not in df.columns:

        print(
            "warning:",
            article_weight_col,
            "not found; use 1.0"
        )

        df[
            article_weight_col
        ] = 1.0

    # ======================================================
    # 数值清理
    # ======================================================

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

    for col in [
        "confidence",
        "relevance",
        "policy_strength",
    ]:

        df[col] = (
            df[col]
            .fillna(0.0)
            .clip(
                0.0,
                1.0
            )
        )

    df["effective_direction"] = (
        df["effective_direction"]
        .fillna(0)
        .astype(int)
    )

    df["policy_type"] = (
        df["policy_type"]
        .fillna("other")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # ======================================================
    # AI质量权重
    # ======================================================

    agg_cfg = cfg.get(
        "aggregation",
        {}
    )

    df["ai_weight"] = (
        df[
            article_weight_col
        ].astype(float)
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

    # ======================================================
    # 基础 signed signal
    # ======================================================

    df["positive_weight"] = np.where(
        df["effective_direction"] == 1,
        df["ai_weight"],
        0.0,
    )

    # 注意：negative保存正数magnitude
    df["negative_weight"] = np.where(
        df["effective_direction"] == -1,
        df["ai_weight"],
        0.0,
    )

    df["signed_weight"] = (
        df["ai_weight"]
        * df["effective_direction"]
    )

    # ======================================================
    # E：quality signal
    #
    # 不使用direction，只保留：
    # AI relevance/confidence/policy strength
    # ======================================================

    df["quality_weight"] = (
        df["ai_weight"]
    )

    # ======================================================
    # G：高置信负面事件
    # ======================================================

    high_negative_mask = (
        (df["effective_direction"] == -1)
        &
        (
            df["confidence"]
            >= high_confidence
        )
        &
        (
            df["relevance"]
            >= high_relevance
        )
        &
        (
            df["policy_strength"]
            >= high_strength
        )
    )

    df[
        "high_conf_negative_weight"
    ] = np.where(
        high_negative_mask,
        df["ai_weight"],
        0.0,
    )

    # ======================================================
    # H：监管/限制/风险类政策
    #
    # 保留signed direction
    # ======================================================

    constraint_mask = (
        df["policy_type"]
        .isin(
            constraint_types
        )
    )

    df[
        "constraint_signed_weight"
    ] = np.where(
        constraint_mask,
        df["signed_weight"],
        0.0,
    )

    # ======================================================
    # denominator：
    # 每天人民日报全部文章数量
    # ======================================================

    news_path = inp[
        "news_path"
    ]

    print(
        "[2] loading news denominator:",
        news_path
    )

    news = pd.read_parquet(
        news_path
    )

    news_date_col = find_date_column(
        news,
        configured_date_col
    )

    news["date"] = (
        pd.to_datetime(
            news[
                news_date_col
            ],
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

    # ======================================================
    # 日期 × framework_code
    # ======================================================

    print(
        "[3] aggregating date x industry..."
    )

    valid = df[
        df["date"].notna()
        &
        df[
            industry_code_col
        ].notna()
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

            quality_weight=(
                "quality_weight",
                "sum"
            ),

            high_conf_negative_weight=(
                "high_conf_negative_weight",
                "sum"
            ),

            constraint_signed_weight=(
                "constraint_signed_weight",
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

    denominator = (
        daily[
            "total_articles"
        ]
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

    daily[
        "quality_attention"
    ] = (
        daily["quality_weight"]
        / denominator
    )

    daily[
        "high_conf_negative_attention"
    ] = (
        daily[
            "high_conf_negative_weight"
        ]
        / denominator
    )

    daily[
        "constraint_attention"
    ] = (
        daily[
            "constraint_signed_weight"
        ]
        / denominator
    )

    # ======================================================
    # 行业代码
    # ======================================================

    daily["industry"] = (
        pd.to_numeric(
            daily[
                industry_code_col
            ],
            errors="coerce"
        )
        .astype("Int64")
        .astype(str)
        .str.zfill(6)
    )

    if (
        industry_code_col
        != "industry"
    ):

        daily = daily.drop(
            columns=[
                industry_code_col
            ]
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

    # ======================================================
    # diagnostics
    # ======================================================

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
        "industries:",
        daily[
            "industry"
        ].nunique()
    )

    print("")

    signal_cols = [
        "positive_attention",
        "negative_attention",
        "net_attention",
        "quality_attention",
        "high_conf_negative_attention",
        "constraint_attention",
    ]

    print(
        daily[
            signal_cols
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
        .value_counts()
        .sort_index()
    )

    print("")
    print(
        "high confidence negative:",
        int(
            high_negative_mask.sum()
        )
    )

    print("")
    print(
        "policy types:"
    )

    print(
        df[
            "policy_type"
        ]
        .value_counts()
        .head(20)
    )

    print("=" * 70)

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