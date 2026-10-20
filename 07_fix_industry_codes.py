# -*- coding: utf-8 -*-

"""
不用重新跑全文分类。

直接根据 industry 名称，
修正 article_industry.parquet 中的：
    framework_code
    index_code
"""

import pandas as pd

from config import PROCESSED_DIR
from sw1 import INDUSTRY_BY_NAME


INPUT_FILE = (
    PROCESSED_DIR
    / "article_industry.parquet"
)

BACKUP_FILE = (
    PROCESSED_DIR
    / "article_industry_before_code_fix.parquet"
)


def main():

    df = pd.read_parquet(
        INPUT_FILE
    )

    print(
        "[rows]",
        f"{len(df):,}"
    )

    # 备份
    df.to_parquet(
        BACKUP_FILE,
        index=False
    )

    unknown = sorted(
        set(df["industry"].unique())
        -
        set(INDUSTRY_BY_NAME.keys())
    )

    if unknown:
        raise ValueError(
            "unknown industries: "
            + str(unknown)
        )

    df["framework_code"] = (
        df["industry"]
        .map(
            lambda x:
            INDUSTRY_BY_NAME[x].framework_code
        )
    )

    df["index_code"] = (
        df["industry"]
        .map(
            lambda x:
            INDUSTRY_BY_NAME[x].index_code
        )
    )

    df.to_parquet(
        INPUT_FILE,
        index=False
    )

    print()
    print(
        df[
            [
                "industry",
                "framework_code",
                "index_code",
            ]
        ]
        .drop_duplicates()
        .sort_values("framework_code")
        .to_string(index=False)
    )

    print()
    print(
        "[backup]",
        BACKUP_FILE
    )

    print(
        "[done]",
        INPUT_FILE
    )


if __name__ == "__main__":
    main()