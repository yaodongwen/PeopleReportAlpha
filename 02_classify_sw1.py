# -*- coding: utf-8 -*-

"""
Step 2

人民日报文章
-> 申万一级行业高置信分类
"""

import pandas as pd

from config import PROCESSED_DIR
from classifier import IndustryClassifier


NEWS_FILE = (
    PROCESSED_DIR
    / "news_clean.parquet"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "article_industry.parquet"
)


def classify_article(
    classifier,
    row,
):

    matches = classifier.classify(
        title=row.title,
        text=row.text,
    )

    output = []

    for match in matches:

        output.append({

            "article_id":
                row.article_id,

            "date":
                row.date,

            "page_num":
                row.page_num,

            "title":
                row.title,

            "industry":
                match.name,

            "framework_code":
                match.framework_code,

            "index_code":
                match.index_code,

            "title_terms":
                "|".join(
                    match.title_terms
                ),

            "body_terms":
                "|".join(
                    match.body_terms
                ),

            "context_terms":
                "|".join(
                    match.context_terms
                ),

            "title_hits":
                match.title_hits,

            "body_hits":
                match.body_hits,

            "score":
                match.score,

            "reason":
                match.reason,

            "article_weight":
                match.weight,
        })

    return output


def main():

    if not NEWS_FILE.exists():
        raise FileNotFoundError(
            NEWS_FILE
        )

    news = pd.read_parquet(
        NEWS_FILE
    )

    classifier = (
        IndustryClassifier()
    )

    output = []

    total = len(news)

    for i, row in enumerate(
        news.itertuples(index=False),
        start=1
    ):

        output.extend(
            classify_article(
                classifier,
                row,
            )
        )

        if i % 10000 == 0:

            print(
                f"[classify] "
                f"{i:,} / {total:,}"
            )

    result = pd.DataFrame(
        output
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    n_articles = (
        result["article_id"]
        .nunique()
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION SUMMARY V2")
    print("=" * 70)

    print(
        "all articles:",
        f"{len(news):,}"
    )

    print(
        "classified articles:",
        f"{n_articles:,}"
    )

    print(
        "coverage:",
        f"{n_articles / len(news):.2%}"
    )

    print(
        "article-industry rows:",
        f"{len(result):,}"
    )

    print(
        "avg industries:",
        f"{len(result) / n_articles:.3f}"
    )

    print()
    print(
        "[done]",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()