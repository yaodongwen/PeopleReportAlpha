# -*- coding: utf-8 -*-

import gzip
import json
import re
from pathlib import Path

import pandas as pd


def clean_text(value):
    """统一清洗文本。"""

    if value is None:
        return ""

    text = str(value)

    text = text.replace("\u3000", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def parse_date(value):
    """把各种日期格式统一成 pandas Timestamp。"""

    if value is None:
        return pd.NaT

    return pd.to_datetime(
        value,
        errors="coerce"
    )


def parse_page_num(value):
    """
    从版面字段中提取数字。

    示例：
        第01版 -> 1
        第2版  -> 2
        03     -> 3
    """

    if value is None:
        return None

    match = re.search(
        r"\d+",
        str(value)
    )

    if not match:
        return None

    return int(
        match.group()
    )


def normalize_record(record):
    """统一不同年份新闻字段。"""

    return {
        "date": parse_date(
            record.get("date")
        ),

        "title": clean_text(
            record.get("title")
        ),

        "author": clean_text(
            record.get("author")
        ),

        "page": clean_text(
            record.get("page")
        ),

        "page_num": parse_page_num(
            record.get("page")
        ),

        "text": clean_text(
            record.get("text")
        ),

        "url": clean_text(
            record.get("url")
        ),
    }


def load_raw_news(path):
    """
    读取单个 .jsonl.gz 文件。
    """

    path = Path(path)

    rows = []

    with gzip.open(
        path,
        "rt",
        encoding="utf-8"
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                print(
                    f"[warn] bad json: "
                    f"{path.name}:{line_no}"
                )
                continue

            rows.append(
                normalize_record(record)
            )

    return pd.DataFrame(rows)


def save_processed(
    df,
    output_path
):
    """保存 parquet。"""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        output_path,
        index=False
    )