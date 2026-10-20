# -*- coding: utf-8 -*-

"""
Step 1

读取人民日报原始 jsonl.gz
→ 清洗
→ 合并
→ 保存 news_clean.parquet
"""

import pandas as pd

from config import (
    RAW_DIR,
    PROCESSED_DIR,
)

from news_io import (
    load_raw_news,
    save_processed,
)


OUTPUT_FILE = (
    PROCESSED_DIR
    / "news_clean.parquet"
)


def get_raw_files():
    """获取所有年度原始文件。"""

    return sorted(
        RAW_DIR.glob(
            "*.jsonl.gz"
        )
    )


def clean_dataframe(df):
    """做 DataFrame 层面的清洗。"""

    if df.empty:
        return df

    # 去掉无日期记录
    df = df[
        df["date"].notna()
    ].copy()

    # 统一日期到天
    df["date"] = (
        df["date"]
        .dt.normalize()
    )

    # 没标题且没正文的数据没有意义
    df = df[
        (df["title"] != "")
        |
        (df["text"] != "")
    ].copy()

    return df


def load_all_news(
    raw_files
):

    parts = []

    for path in raw_files:

        print(
            f"[read] {path.name}"
        )

        df = load_raw_news(
            path
        )

        df = clean_dataframe(
            df
        )

        print(
            f"       rows={len(df):,}"
        )

        if not df.empty:
            parts.append(
                df
            )

    if not parts:
        raise RuntimeError(
            "没有读取到有效新闻数据"
        )

    return pd.concat(
        parts,
        ignore_index=True
    )


def finalize_news(df):

    df = df.copy()

    # 排序
    df = df.sort_values(
        [
            "date",
            "page_num",
            "title",
        ],
        na_position="last"
    ).reset_index(
        drop=True
    )

    # 文章唯一ID
    df.insert(
        0,
        "article_id",
        range(
            len(df)
        )
    )

    return df


def print_summary(df):

    print()
    print("=" * 70)
    print("NEWS SUMMARY")
    print("=" * 70)

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
        "days:",
        df["date"].nunique()
    )

    print()

    print(
        df[
            [
                "article_id",
                "date",
                "page_num",
                "title",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


def main():

    raw_files = get_raw_files()

    if not raw_files:
        raise FileNotFoundError(
            RAW_DIR
        )

    print(
        f"Found {len(raw_files)} raw files"
    )

    news = load_all_news(
        raw_files
    )

    news = finalize_news(
        news
    )

    save_processed(
        news,
        OUTPUT_FILE
    )

    print_summary(
        news
    )

    print()
    print(
        "[done]",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()