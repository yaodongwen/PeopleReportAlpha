# -*- coding: utf-8 -*-

from pathlib import Path


BASE_DIR = Path(
    "/Users/yaodongdong/WorkPlace/intern/PeopleDailyFactor"
)

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FACTOR_DIR = BASE_DIR / "data" / "factor"


for path in [
    RAW_DIR,
    PROCESSED_DIR,
    FACTOR_DIR,
]:
    path.mkdir(
        parents=True,
        exist_ok=True
    )


NEWS_FILE = (
    PROCESSED_DIR
    / "news_clean.parquet"
)

ARTICLE_INDUSTRY_FILE = (
    PROCESSED_DIR
    / "article_industry.parquet"
)

DAILY_SIGNAL_FILE = (
    FACTOR_DIR
    / "industry_daily.parquet"
)


# ============================================================
# 使用区间
# ============================================================

START_YEAR = 2000
END_YEAR = 2025


# ============================================================
# 分类参数
# ============================================================

TITLE_WEIGHT = 3.0
BODY_WEIGHT = 1.0

# 一篇文章至少达到这个分数，才认为属于该行业
MIN_INDUSTRY_SCORE = 3.0

# 防止一篇宏观文章同时被打到十几个行业
MAX_INDUSTRIES_PER_ARTICLE = 4