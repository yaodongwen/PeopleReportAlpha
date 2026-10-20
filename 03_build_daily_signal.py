# -*- coding: utf-8 -*-

"""
Step 3

article_industry.parquet
+
news_clean.parquet

->

按“日期 × 申万一级行业”
生成日度政策关注信号。
"""

import pandas as pd

from config import (
    PROCESSED_DIR,
    FACTOR_DIR,
)

from signal_builder import (
    build_daily_signal,
)


NEWS_FILE = (
    PROCESSED_DIR
    / "news_clean.parquet"
)

ARTICLE_INDUSTRY_FILE = (
    PROCESSED_DIR
    / "article_industry.parquet"
)

OUTPUT_FILE = (
    FACTOR_DIR
    / "industry_daily.parquet"
)


def load_data():

    if not NEWS_FILE.exists():
        raise FileNotFoundError(
            NEWS_FILE
        )

    if not ARTICLE_INDUSTRY_FILE.exists():
        raise FileNotFoundError(
            ARTICLE_INDUSTRY_FILE
        )

    news = pd.read_parquet(
        NEWS_FILE
    )

    classified = pd.read_parquet(
        ARTICLE_INDUSTRY_FILE
    )

    return news, classified


def print_summary(daily):

    print()
    print("=" * 70)
    print("DAILY INDUSTRY SIGNAL SUMMARY")
    print("=" * 70)

    print(
        "rows:",
        f"{len(daily):,}"
    )

    print(
        "date:",
        daily["date"].min(),
        "->",
        daily["date"].max()
    )

    print(
        "days:",
        daily["date"].nunique()
    )

    print(
        "industries:",
        daily["industry"].nunique()
    )

    print()

    print(
        daily[
            [
                "date",
                "industry",
                "article_count",
                "weighted_count",
                "total_articles",
                "attention",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


def main():

    news, classified = load_data()

    print(
        "[news]",
        f"{len(news):,}"
    )

    print(
        "[article-industry]",
        f"{len(classified):,}"
    )

    daily = build_daily_signal(
        news=news,
        classified=classified,
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

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    daily.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print_summary(
        daily
    )

    print()
    print(
        "[done]",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
