# -*- coding: utf-8 -*-

"""
Step 4

检查人民日报 -> 申万一级行业分类质量。

输出：
1. 各行业文章数量
2. 缺失行业
3. 多行业文章比例
4. 每个行业随机抽样文章
"""

import pandas as pd

from config import PROCESSED_DIR
from sw1 import INDUSTRIES


NEWS_FILE = (
    PROCESSED_DIR
    / "news_clean.parquet"
)

CLASSIFIED_FILE = (
    PROCESSED_DIR
    / "article_industry.parquet"
)


SAMPLE_INDUSTRIES = [
    "银行",
    "汽车",
    "电力设备",
    "公用事业",
    "交通运输",
    "计算机",
    "有色金属",
    "房地产",
]

SAMPLE_N = 20


def load_data():

    news = pd.read_parquet(
        NEWS_FILE
    )

    classified = pd.read_parquet(
        CLASSIFIED_FILE
    )

    return news, classified


def print_basic_summary(
    news,
    classified
):

    n_news = len(news)

    n_classified = (
        classified["article_id"]
        .nunique()
    )

    coverage = (
        n_classified
        / n_news
    )

    avg_industry = (
        len(classified)
        / n_classified
    )

    print("=" * 80)
    print("BASIC SUMMARY")
    print("=" * 80)

    print(
        f"all articles       : {n_news:,}"
    )

    print(
        f"classified articles: {n_classified:,}"
    )

    print(
        f"coverage           : {coverage:.2%}"
    )

    print(
        f"article-industry   : {len(classified):,}"
    )

    print(
        f"avg industries     : {avg_industry:.3f}"
    )


def print_industry_counts(
    classified
):

    counts = (
        classified
        .groupby("industry")
        .agg(
            articles=(
                "article_id",
                "nunique"
            ),
            rows=(
                "article_id",
                "size"
            ),
            avg_score=(
                "score",
                "mean"
            ),
        )
        .sort_values(
            "articles",
            ascending=False
        )
    )

    print()
    print("=" * 80)
    print("INDUSTRY COUNTS")
    print("=" * 80)

    print(
        counts.to_string()
    )

    return counts


def print_missing_industries(
    classified
):

    expected = {
        x.name
        for x in INDUSTRIES
    }

    actual = set(
        classified["industry"]
        .unique()
    )

    missing = sorted(
        expected - actual
    )

    print()
    print("=" * 80)
    print("MISSING INDUSTRIES")
    print("=" * 80)

    print(
        missing
    )


def print_multi_label_stats(
    classified
):

    n = (
        classified
        .groupby("article_id")
        .size()
    )

    print()
    print("=" * 80)
    print("MULTI-LABEL")
    print("=" * 80)

    print(
        n.value_counts()
        .sort_index()
        .to_string()
    )

    print()

    for k in range(
        1,
        int(n.max()) + 1
    ):

        ratio = (
            n.eq(k)
            .mean()
        )

        print(
            f"{k} industries: "
            f"{ratio:.2%}"
        )


def sample_industry(
    classified,
    industry,
    n=SAMPLE_N
):

    x = classified[
        classified["industry"]
        == industry
    ].copy()

    if x.empty:
        return

    n = min(
        n,
        len(x)
    )

    sample = x.sample(
        n=n,
        random_state=42
    )

    cols = [
        "date",
        "score",
        "title_hits",
        "body_hits",
        "article_weight",
        "title",
    ]

    print()
    print("=" * 100)
    print(
        f"SAMPLE: {industry}"
    )
    print("=" * 100)

    print(
        sample[cols]
        .sort_values(
            "date"
        )
        .to_string(
            index=False
        )
    )


def main():

    news, classified = (
        load_data()
    )

    print_basic_summary(
        news,
        classified
    )

    print_industry_counts(
        classified
    )

    print_missing_industries(
        classified
    )

    print_multi_label_stats(
        classified
    )

    for industry in SAMPLE_INDUSTRIES:

        sample_industry(
            classified,
            industry
        )


if __name__ == "__main__":
    main()