# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ============================================================
# Cross-sectional rank
# ============================================================

def cross_sectional_rank(series):

    valid = (
        series.notna()
        & np.isfinite(series)
    )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=np.float64,
    )

    n = int(
        valid.sum()
    )

    if n <= 1:
        return result

    ranked = (
        series.loc[valid]
        .rank(
            method="average",
            pct=True,
        )
    )

    result.loc[valid] = (
        ranked * 2.0 - 1.0
    )

    return result


# ============================================================
# Surprise
# ============================================================

def calc_surprise(
    x,
    history_window,
    min_history,
    std_floor,
):

    historical = (
        x.shift(1)
    )

    mean = (
        historical
        .rolling(
            history_window,
            min_periods=min_history,
        )
        .mean()
    )

    std = (
        historical
        .rolling(
            history_window,
            min_periods=min_history,
        )
        .std(ddof=1)
    )

    # 避免 -658 这类 tiny std 爆炸
    std = std.where(
        std > std_floor,
        np.nan
    )

    return (
        (x - mean)
        / std
    )


# ============================================================
# Full calendar
# ============================================================

def build_full_calendar_panel(
    df
):

    df = df.copy()

    df["date"] = (
        pd.to_datetime(
            df["date"]
        )
    )

    df["industry"] = (
        df["industry"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False
        )
        .str.zfill(6)
    )

    dates = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="D",
    )

    industries = sorted(
        df[
            "industry"
        ]
        .dropna()
        .unique()
    )

    full_index = (
        pd.MultiIndex.from_product(
            [
                dates,
                industries,
            ],
            names=[
                "date",
                "industry",
            ],
        )
    )

    value_cols = [
        "positive_attention",
        "negative_attention",
        "net_attention",
        "quality_attention",
        "high_conf_negative_attention",
        "constraint_attention",
    ]

    missing = [
        c
        for c in value_cols
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            "Run updated build_signed_daily.py first. "
            "Missing columns: %s"
            % missing
        )

    panel = (
        df[
            [
                "date",
                "industry",
            ]
            + value_cols
        ]
        .groupby(
            [
                "date",
                "industry",
            ],
            as_index=False,
        )[value_cols]
        .sum()
        .set_index(
            [
                "date",
                "industry",
            ]
        )
        .reindex(
            full_index
        )
    )

    panel[
        value_cols
    ] = (
        panel[
            value_cols
        ]
        .fillna(0.0)
    )

    return (
        panel.reset_index()
    )


# ============================================================
# Industry time series
# ============================================================

def add_industry_factors(
    g,
    history_window,
    min_history,
    std_floor,
    ewm_halflife,
):

    g = (
        g
        .sort_values("date")
        .copy()
    )

    # ======================================================
    # 原有因子
    # ======================================================

    g["positive_20"] = (
        g[
            "positive_attention"
        ]
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
    )

    g["negative_20"] = (
        g[
            "negative_attention"
        ]
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
    )

    g["net_20"] = (
        g[
            "net_attention"
        ]
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
    )

    # ======================================================
    # A/C：10日 / 40日 net
    # ======================================================

    g["net_10"] = (
        g[
            "net_attention"
        ]
        .rolling(
            10,
            min_periods=1,
        )
        .sum()
    )

    g["net_40"] = (
        g[
            "net_attention"
        ]
        .rolling(
            40,
            min_periods=1,
        )
        .sum()
    )

    # ======================================================
    # D：指数衰减
    # ======================================================

    g["net_ewm10"] = (
        g[
            "net_attention"
        ]
        .ewm(
            halflife=ewm_halflife,
            adjust=False,
        )
        .mean()
    )

    # ======================================================
    # E：AI质量，不考虑方向
    # ======================================================

    g["quality_20"] = (
        g[
            "quality_attention"
        ]
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
    )

    # ======================================================
    # G：高置信负面事件
    # ======================================================

    g[
        "high_conf_negative_20"
    ] = (
        g[
            "high_conf_negative_attention"
        ]
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
    )

    # ======================================================
    # H：监管/限制/风险
    # ======================================================

    g["constraint_20"] = (
        g[
            "constraint_attention"
        ]
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
    )

    # ======================================================
    # Surprise
    # ======================================================

    surprise_sources = {
        "positive_surprise_20":
            "positive_20",

        "negative_surprise_20":
            "negative_20",

        "net_surprise_20":
            "net_20",

        "net_surprise_10":
            "net_10",

        "net_surprise_40":
            "net_40",

        "net_ewm10_surprise":
            "net_ewm10",

        "quality_surprise_20":
            "quality_20",

        "constraint_surprise_20":
            "constraint_20",
    }

    for output_col, source_col in (
        surprise_sources.items()
    ):

        g[output_col] = (
            calc_surprise(
                g[source_col],
                history_window,
                min_history,
                std_floor,
            )
        )

    return g


# ============================================================
# Original PeopleDaily factor
# ============================================================

def load_original_factor_npz(
    path
):

    print(
        "[original factor]",
        path
    )

    x = np.load(
        path
    )

    dates = (
        x["dates"]
        .astype(np.int64)
    )

    codes = (
        x["codes"]
        .astype(str)
    )

    matrix = (
        x[
            "surprise_20_rank"
        ]
        .astype(np.float64)
    )

    records = []

    for j, code in enumerate(
        codes
    ):

        tmp = pd.DataFrame(
            {
                "date_int":
                    dates,

                "industry":
                    str(code).zfill(6),

                "original_surprise_20_rank":
                    matrix[:, j],
            }
        )

        records.append(
            tmp
        )

    return pd.concat(
        records,
        ignore_index=True
    )


# ============================================================
# Daily residual
# ============================================================

def residualize_one_date(
    g
):

    result = pd.Series(
        np.nan,
        index=g.index,
        dtype=np.float64,
    )

    x = (
        g[
            "original_surprise_20_rank"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    y = (
        g[
            "net_surprise_20_rank"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if int(
        valid.sum()
    ) < 5:

        return result

    X = np.column_stack(
        [
            np.ones(
                int(
                    valid.sum()
                )
            ),
            x[valid],
        ]
    )

    beta = np.linalg.lstsq(
        X,
        y[valid],
        rcond=None,
    )[0]

    residual = (
        y[valid]
        -
        X.dot(beta)
    )

    result.loc[
        g.index[valid]
    ] = residual

    return result


# ============================================================
# Main factor construction
# ============================================================

def build_factors(
    df,
    history_window,
    min_history,
    std_floor,
    ewm_halflife,
    original_npz,
):

    print(
        "[1] building full calendar panel..."
    )

    factor = (
        build_full_calendar_panel(
            df
        )
    )

    print(
        "calendar rows:",
        len(factor)
    )

    print(
        "date range:",
        factor["date"].min(),
        "->",
        factor["date"].max()
    )

    print(
        "industries:",
        factor[
            "industry"
        ].nunique()
    )

    print(
        "[2] calculating rolling factors..."
    )

    pieces = []

    for industry, g in factor.groupby(
        "industry",
        sort=False,
    ):

        pieces.append(
            add_industry_factors(
                g,
                history_window,
                min_history,
                std_floor,
                ewm_halflife,
            )
        )

    factor = pd.concat(
        pieces,
        ignore_index=True,
    )

    factor = (
        factor
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
    # Rank
    # ======================================================

    print(
        "[3] calculating cross-sectional ranks..."
    )

    rank_sources = {
        # 原有
        "positive_attention_20_rank":
            "positive_20",

        "negative_attention_20_rank":
            "negative_20",

        "net_attention_20_rank":
            "net_20",

        "positive_surprise_20_rank":
            "positive_surprise_20",

        "negative_surprise_20_rank":
            "negative_surprise_20",

        "net_surprise_20_rank":
            "net_surprise_20",

        # A
        "net_surprise_10_rank":
            "net_surprise_10",

        # C
        "net_surprise_40_rank":
            "net_surprise_40",

        # D
        "net_ewm10_surprise_rank":
            "net_ewm10_surprise",

        # E
        "quality_surprise_20_rank":
            "quality_surprise_20",

        # G
        "high_conf_negative_20_rank":
            "high_conf_negative_20",

        # H
        "constraint_surprise_20_rank":
            "constraint_surprise_20",
    }

    for output_col, source_col in (
        rank_sources.items()
    ):

        factor[
            output_col
        ] = (
            factor
            .groupby(
                "date",
                group_keys=False,
            )[source_col]
            .apply(
                cross_sectional_rank
            )
        )

    # ======================================================
    # date int
    # ======================================================

    factor["date_int"] = (
        factor["date"]
        .dt.strftime(
            "%Y%m%d"
        )
        .astype(np.int64)
    )

    # ======================================================
    # F：AI residual
    #
    # AI net surprise剔除原始PeopleDaily attention surprise
    # ======================================================

    if (
        original_npz
        and Path(
            original_npz
        ).exists()
    ):

        print(
            "[4] calculating AI residual factor..."
        )

        original = (
            load_original_factor_npz(
                original_npz
            )
        )

        factor = factor.merge(
            original,
            on=[
                "date_int",
                "industry",
            ],
            how="left",
            validate="one_to_one",
        )

        residual = (
            factor
            .groupby(
                "date",
                group_keys=False,
            )
            .apply(
                residualize_one_date
            )
        )

        # pandas不同版本下groupby.apply索引可能不同，
        # 所以按照原index赋值
        if isinstance(
            residual.index,
            pd.MultiIndex
        ):

            residual.index = (
                residual.index
                .get_level_values(-1)
            )

        factor[
            "ai_residual_raw"
        ] = (
            residual
            .reindex(
                factor.index
            )
        )

        factor[
            "ai_residual_rank"
        ] = (
            factor
            .groupby(
                "date",
                group_keys=False,
            )[
                "ai_residual_raw"
            ]
            .apply(
                cross_sectional_rank
            )
        )

    else:

        print(
            "[4] original NPZ not found; "
            "ai_residual_rank will be NaN"
        )

        factor[
            "original_surprise_20_rank"
        ] = np.nan

        factor[
            "ai_residual_raw"
        ] = np.nan

        factor[
            "ai_residual_rank"
        ] = np.nan

    return factor


# ============================================================
# Diagnostics
# ============================================================

def print_diagnostics(
    df
):

    print("")
    print("=" * 72)
    print("AI POLICY FACTOR DIAGNOSTICS")
    print("=" * 72)

    print(
        "rows:",
        len(df)
    )

    print(
        "dates:",
        df[
            "date"
        ].nunique()
    )

    print(
        "industries:",
        df[
            "industry"
        ].nunique()
    )

    test_cols = [
        "net_surprise_10_rank",
        "net_surprise_20_rank",
        "net_surprise_40_rank",
        "net_ewm10_surprise_rank",
        "quality_surprise_20_rank",
        "ai_residual_rank",
        "high_conf_negative_20_rank",
        "constraint_surprise_20_rank",
    ]

    print("")
    print(
        df[
            test_cols
        ]
        .describe()
        .T[
            [
                "count",
                "mean",
                "std",
                "min",
                "50%",
                "max",
            ]
        ]
    )

    print("")
    print(
        "factor correlation:"
    )

    print(
        df[
            test_cols
        ]
        .corr()
        .round(4)
    )

    print("=" * 72)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:

        cfg = yaml.safe_load(f)

    ai_cfg = cfg.get(
        "ai_factor",
        {}
    )

    input_path = (
        ai_cfg.get(
            "input_daily"
        )
        or cfg["output"][
            "daily_parquet"
        ]
    )

    output_path = (
        ai_cfg.get(
            "output_parquet",
            "../data/factor/"
            "people_daily_ai_policy_factor.parquet"
        )
    )

    original_npz = (
        ai_cfg.get(
            "original_factor_npz",
            "../data/factor/"
            "people_daily_industry_factor.npz"
        )
    )

    history_window = int(
        ai_cfg.get(
            "history_window",
            365
        )
    )

    min_history = int(
        ai_cfg.get(
            "min_history",
            120
        )
    )

    std_floor = float(
        ai_cfg.get(
            "std_floor",
            1e-6
        )
    )

    ewm_halflife = float(
        ai_cfg.get(
            "ewm_halflife",
            10
        )
    )

    print(
        "input:",
        input_path
    )

    print(
        "output:",
        output_path
    )

    print(
        "original factor:",
        original_npz
    )

    print(
        "history:",
        history_window
    )

    print(
        "minHistory:",
        min_history
    )

    print(
        "stdFloor:",
        std_floor
    )

    print(
        "ewmHalflife:",
        ewm_halflife
    )

    df = pd.read_parquet(
        input_path
    )

    factor = build_factors(
        df,
        history_window,
        min_history,
        std_floor,
        ewm_halflife,
        original_npz,
    )

    print_diagnostics(
        factor
    )

    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    factor.to_parquet(
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
    