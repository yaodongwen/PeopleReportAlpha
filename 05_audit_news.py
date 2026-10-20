# -*- coding: utf-8 -*-

import pandas as pd
from config import PROCESSED_DIR


NEWS_FILE = PROCESSED_DIR / "news_clean.parquet"


def main():

    df = pd.read_parquet(NEWS_FILE)

    df["title_len"] = (
        df["title"]
        .fillna("")
        .str.len()
    )

    df["text_len"] = (
        df["text"]
        .fillna("")
        .str.len()
    )

    print("=" * 80)
    print("TITLE LENGTH")
    print("=" * 80)

    print(
        df["title_len"]
        .describe(
            percentiles=[
                0.5,
                0.9,
                0.95,
                0.99,
                0.999,
            ]
        )
    )

    print()
    print("=" * 80)
    print("LONG TITLES")
    print("=" * 80)

    long_title = (
        df[df["title_len"] > 200]
        .sort_values(
            "title_len",
            ascending=False
        )
    )

    print(
        long_title[
            [
                "article_id",
                "date",
                "title_len",
                "text_len",
                "title",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print()
    print(
        "title > 100:",
        (df["title_len"] > 100).sum()
    )

    print(
        "title > 200:",
        (df["title_len"] > 200).sum()
    )

    print(
        "title > 500:",
        (df["title_len"] > 500).sum()
    )


if __name__ == "__main__":
    main()