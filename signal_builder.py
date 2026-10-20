# -*- coding: utf-8 -*-

"""
把“文章 -> 行业”的分类结果
聚合成“日期 -> 行业”的日度政策关注信号。
"""

import pandas as pd


def page_weight(page_num):
    """
    人民日报版面权重。

    第1版最重要，
    2~3版次之，
    4~8版稍低，
    其他版面权重为1。
    """

    if pd.isna(page_num):
        return 1.0

    page_num = int(page_num)

    if page_num == 1:
        return 2.0

    if page_num <= 3:
        return 1.5

    if page_num <= 8:
        return 1.2

    return 1.0


def add_article_weight(classified):
    """
    给每条“文章-行业”记录增加最终权重。

    article_weight:
        一篇文章涉及多个行业时，
        分类器已经把总权重归一到1。

    page_weight:
        版面越重要，权重越高。
    """

    df = classified.copy()

    df["page_weight"] = (
        df["page_num"]
        .map(page_weight)
    )

    df["weighted_article"] = (
        df["article_weight"]
        * df["page_weight"]
    )

    return df


def build_daily_total(news):
    """
    计算人民日报每天总文章数。

    后面行业文章数量除以当天总文章数，
    用来消除不同时期报纸文章总量差异。
    """

    return (
        news
        .groupby("date")
        .size()
        .rename("total_articles")
        .reset_index()
    )


def aggregate_industry_daily(classified):
    """
    聚合成：
        日期 × 行业
    """

    return (
        classified
        .groupby(
            [
                "date",
                "industry",
                "framework_code",
                "index_code",
            ],
            as_index=False
        )
        .agg(
            article_count=(
                "article_id",
                "nunique"
            ),

            weighted_count=(
                "weighted_article",
                "sum"
            ),

            title_hits=(
                "title_hits",
                "sum"
            ),

            body_hits=(
                "body_hits",
                "sum"
            ),
        )
    )


def build_daily_signal(
    news,
    classified
):
    """
    构建日度行业政策关注度。

    attention =
        行业加权文章数
        /
        当天人民日报总文章数
    """

    classified = add_article_weight(
        classified
    )

    daily_total = build_daily_total(
        news
    )

    daily = aggregate_industry_daily(
        classified
    )

    daily = daily.merge(
        daily_total,
        on="date",
        how="left"
    )

    daily["attention"] = (
        daily["weighted_count"]
        / daily["total_articles"]
    )

    return daily
