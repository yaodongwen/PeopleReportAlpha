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

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=np.float64,
    )

    valid = (
        series.notna()
        & np.isfinite(series)
    )

    n = int(valid.sum())

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
# Time-series surprise
# ============================================================

def calc_surprise(
    x,
    history_window,
    min_history,
    std_floor,
):

    historical = x.shift(1)

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

    # 防止极小 std 造成极端 z-score
    std = std.where(
        std > std_floor,
        np.nan
    )

    return (
        (x - mean)
        / std
    )


# ============================================================
# Full natural-calendar panel
# ============================================================

def build_full_calendar_panel(df):

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
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
        df["industry"]
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
            "Missing daily columns: %s. "
            "Run updated build_signed_daily.py first."
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

    # 没新闻 -> 0
    panel[
        value_cols
    ] = (
        panel[
            value_cols
        ]
        .fillna(0.0)
    )

    return panel.reset_index()


# ============================================================
# Per-industry time-series factors
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
    # Positive / Negative / Net
    # ======================================================

    g["positive_20"] = (
        g["positive_attention"]
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
    )

    g["negative_20"] = (
        g["negative_attention"]
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
    )

    g["net_10"] = (
        g["net_attention"]
        .rolling(
            10,
            min_periods=1,
        )
        .sum()
    )

    g["net_20"] = (
        g["net_attention"]
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
    )

    g["net_30"] = (
        g["net_attention"]
        .rolling(
            30,
            min_periods=1,
        )
        .sum()
    )

    g["net_40"] = (
        g["net_attention"]
        .rolling(
            40,
            min_periods=1,
        )
        .sum()
    )

    # ======================================================
    # EWMA net
    # ======================================================

    g["net_ewm10"] = (
        g["net_attention"]
        .ewm(
            halflife=ewm_halflife,
            adjust=False,
        )
        .mean()
    )

    # ======================================================
    # Quality
    # ======================================================

    g["quality_20"] = (
        g["quality_attention"]
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
    )

    g["quality_30"] = (
        g["quality_attention"]
        .rolling(
            30,
            min_periods=1,
        )
        .sum()
    )

    g["quality_40"] = (
        g["quality_attention"]
        .rolling(
            40,
            min_periods=1,
        )
        .sum()
    )

    # ======================================================
    # High-confidence negative
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
    # Constraint
    # ======================================================

    g["constraint_20"] = (
        g["constraint_attention"]
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

        "net_surprise_10":
            "net_10",

        "net_surprise_20":
            "net_20",

        "net_surprise_30":
            "net_30",

        "net_surprise_40":
            "net_40",

        "net_ewm10_surprise":
            "net_ewm10",

        "quality_surprise_20":
            "quality_20",

        "quality_surprise_30":
            "quality_30",

        "quality_surprise_40":
            "quality_40",

        "constraint_surprise_20":
            "constraint_20",
    }

    for output_col, source_col in (
        surprise_sources.items()
    ):

        g[output_col] = calc_surprise(
            g[source_col],
            history_window,
            min_history,
            std_floor,
        )

    return g


# ============================================================
# Load original PeopleDaily factor
# ============================================================

def load_original_factor_npz(path):

    print(
        "[original factor]",
        path
    )

    x = np.load(
        path
    )

    if (
        "surprise_20_rank"
        not in x.files
    ):

        raise KeyError(
            "original NPZ does not contain "
            "'surprise_20_rank'. "
            "Available: %s"
            % x.files
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

        clean_code = (
            str(code)
            .replace(
                ".0",
                ""
            )
            .zfill(6)
        )

        tmp = pd.DataFrame(
            {
                "date_int":
                    dates,

                "industry":
                    clean_code,

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
# Cross-sectional residual
# ============================================================

def add_cross_sectional_residual(
    df,
    y_col,
    x_col,
    raw_col,
    rank_col,
):

    residual = pd.Series(
        np.nan,
        index=df.index,
        dtype=np.float64,
    )

    for _, g in df.groupby(
        "date",
        sort=False,
    ):

        idx = g.index

        x = (
            g[x_col]
            .to_numpy(
                dtype=np.float64
            )
        )

        y = (
            g[y_col]
            .to_numpy(
                dtype=np.float64
            )
        )

        valid = (
            np.isfinite(x)
            &
            np.isfinite(y)
        )

        n = int(
            valid.sum()
        )

        if n < 5:
            continue

        X = np.column_stack(
            [
                np.ones(n),
                x[valid],
            ]
        )

        beta = np.linalg.lstsq(
            X,
            y[valid],
            rcond=None,
        )[0]

        resid = (
            y[valid]
            -
            X.dot(beta)
        )

        valid_index = (
            idx[
                np.where(
                    valid
                )[0]
            ]
        )

        residual.loc[
            valid_index
        ] = resid

    df[
        raw_col
    ] = residual

    df[
        rank_col
    ] = (
        df.groupby(
            "date"
        )[raw_col]
        .rank(
            method="average",
            pct=True,
        )
        * 2.0
        - 1.0
    )

    df.loc[
        df[raw_col].isna(),
        rank_col
    ] = np.nan

    return df


# ============================================================
# Add smoothed rank
# ============================================================

def add_smoothed_rank(
    factor,
    source_col,
    halflife,
    raw_col,
    rank_col,
):

    # 必须先按行业、时间排序
    factor = (
        factor
        .sort_values(
            [
                "industry",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    factor[
        raw_col
    ] = (
        factor
        .groupby(
            "industry",
            group_keys=False,
        )[source_col]
        .transform(
            lambda x:
                x.ewm(
                    halflife=halflife,
                    adjust=False,
                    min_periods=1,
                )
                .mean()
        )
    )

    factor[
        rank_col
    ] = (
        factor.groupby(
            "date"
        )[raw_col]
        .rank(
            method="average",
            pct=True,
        )
        * 2.0
        - 1.0
    )

    factor.loc[
        factor[
            raw_col
        ].isna(),
        rank_col
    ] = np.nan

    return factor


# ============================================================
# Main build
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

    # ======================================================
    # Time series
    # ======================================================

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
    # Cross-sectional ranks
    # ======================================================

    print(
        "[3] calculating cross-sectional ranks..."
    )

    rank_sources = {

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

        "net_surprise_10_rank":
            "net_surprise_10",

        "net_surprise_20_rank":
            "net_surprise_20",

        "net_surprise_40_rank":
            "net_surprise_40",

        "net_ewm10_surprise_rank":
            "net_ewm10_surprise",

        "quality_surprise_20_rank":
            "quality_surprise_20",

        "high_conf_negative_20_rank":
            "high_conf_negative_20",

        "constraint_surprise_20_rank":
            "constraint_surprise_20",

        "quality_surprise_30_rank":
            "quality_surprise_30",

        "quality_surprise_40_rank":
            "quality_surprise_40",
    }

    for output_col, source_col in (
        rank_sources.items()
    ):

        factor[
            output_col
        ] = (
            factor.groupby(
                "date"
            )[source_col]
            .rank(
                method="average",
                pct=True,
            )
            * 2.0
            - 1.0
        )

        factor.loc[
            factor[
                source_col
            ].isna(),
            output_col
        ] = np.nan

    # ======================================================
    # date int
    # ======================================================

    factor["date_int"] = (
        factor[
            "date"
        ]
        .dt.strftime(
            "%Y%m%d"
        )
        .astype(
            np.int64
        )
    )

    # ======================================================
    # Original PeopleDaily factor
    # ======================================================

    if (
        original_npz
        and Path(
            original_npz
        ).exists()
    ):

        print(
            "[4] loading original PeopleDaily factor..."
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

    else:

        print(
            "[4] original factor not found:",
            original_npz
        )

        factor[
            "original_surprise_20_rank"
        ] = np.nan

    # ======================================================
    # factor9
    # AI net residual vs original attention
    # ======================================================

    print(
        "[5] factor9: net residual vs original..."
    )

    factor = add_cross_sectional_residual(
        factor,
        y_col="net_surprise_20_rank",
        x_col="original_surprise_20_rank",
        raw_col="ai_residual_raw",
        rank_col="ai_residual_rank",
    )

    # ======================================================
    # factor12
    # direction residual vs AI quality
    # ======================================================

    print(
        "[6] factor12: direction residual..."
    )

    factor = add_cross_sectional_residual(
        factor,
        y_col="net_surprise_20_rank",
        x_col="quality_surprise_20_rank",
        raw_col="direction_residual_raw",
        rank_col="direction_residual_rank",
    )

    # ======================================================
    # factor13
    # quality residual vs original attention
    # ======================================================

    print(
        "[7] factor13: quality residual..."
    )

    factor = add_cross_sectional_residual(
        factor,
        y_col="quality_surprise_20_rank",
        x_col="original_surprise_20_rank",
        raw_col="quality_residual_raw",
        rank_col="quality_residual_rank",
    )

    # ======================================================
    # factor16
    # 20 + 40 net combo
    # ======================================================

    print(
        "[8] factor16: net 20/40 combo..."
    )

    factor[
        "net_surprise_20_40_combo_raw"
    ] = (
        0.50
        * factor[
            "net_surprise_20_rank"
        ]
        +
        0.50
        * factor[
            "net_surprise_40_rank"
        ]
    )

    factor[
        "net_surprise_20_40_combo_rank"
    ] = (
        factor.groupby(
            "date"
        )[
            "net_surprise_20_40_combo_raw"
        ]
        .rank(
            method="average",
            pct=True,
        )
        * 2.0
        - 1.0
    )

    # ======================================================
    # factor17
    # 75% factor16 + 25% factor13
    # ======================================================

    print(
        "[9] factor17: factor16 + quality residual..."
    )

    factor[
        "combo16_quality_residual_raw"
    ] = (
        0.75
        * factor[
            "net_surprise_20_40_combo_rank"
        ]
        +
        0.25
        * factor[
            "quality_residual_rank"
        ]
    )

    factor[
        "combo16_quality_residual_rank"
    ] = (
        factor.groupby(
            "date"
        )[
            "combo16_quality_residual_raw"
        ]
        .rank(
            method="average",
            pct=True,
        )
        * 2.0
        - 1.0
    )

    # ======================================================
    # factor18
    # factor13 EWM5
    # ======================================================

    print(
        "[10] factor18: quality residual EWM5..."
    )

    factor = add_smoothed_rank(
        factor,
        source_col="quality_residual_rank",
        halflife=5,
        raw_col="quality_residual_ewm5_raw",
        rank_col="quality_residual_ewm5_rank",
    )

    # ======================================================
    # factor19
    # factor16 + quality20
    # ======================================================

    print(
        "[11] factor19: factor16 + quality20..."
    )

    factor[
        "combo16_quality20_raw"
    ] = (
        0.50
        * factor[
            "net_surprise_20_40_combo_rank"
        ]
        +
        0.50
        * factor[
            "quality_surprise_20_rank"
        ]
    )

    factor[
        "combo16_quality20_rank"
    ] = (
        factor.groupby(
            "date"
        )[
            "combo16_quality20_raw"
        ]
        .rank(
            method="average",
            pct=True,
        )
        * 2.0
        - 1.0
    )

    # ======================================================
    # factor20
    #
    # factor17 EWM halflife=3
    #
    # 目的：
    # 降噪，但比factor21响应更快
    # ======================================================

    print(
        "[12] factor20: factor17 EWM3..."
    )

    factor = add_smoothed_rank(
        factor,
        source_col="combo16_quality_residual_rank",
        halflife=3,
        raw_col="factor17_ewm3_raw",
        rank_col="factor17_ewm3_rank",
    )

    # ======================================================
    # factor21
    #
    # factor17 EWM halflife=5
    # ======================================================

    print(
        "[13] factor21: factor17 EWM5..."
    )

    factor = add_smoothed_rank(
        factor,
        source_col="combo16_quality_residual_rank",
        halflife=5,
        raw_col="factor17_ewm5_raw",
        rank_col="factor17_ewm5_rank",
    )

    # ======================================================
    # factor22
    #
    # 70% factor17
    # +
    # 30% factor18
    #
    # factor17 = 主信号
    # factor18 = 低换手quality独立增量
    # ======================================================

    print(
        "[14] factor22: factor17 + factor18..."
    )

    factor[
        "combo17_factor18_raw"
    ] = (
        0.70
        * factor[
            "combo16_quality_residual_rank"
        ]
        +
        0.30
        * factor[
            "quality_residual_ewm5_rank"
        ]
    )

    factor[
        "combo17_factor18_rank"
    ] = (
        factor.groupby(
            "date"
        )[
            "combo17_factor18_raw"
        ]
        .rank(
            method="average",
            pct=True,
        )
        * 2.0
        - 1.0
    )

    # ======================================================
    # Restore standard order
    # ======================================================

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

    return factor


# ============================================================
# Diagnostics
# ============================================================

def print_diagnostics(df):

    print("")
    print("=" * 80)
    print("AI POLICY FACTOR DIAGNOSTICS")
    print("=" * 80)

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

    print(
        "date range:",
        df["date"].min(),
        "->",
        df["date"].max()
    )

    test_cols = [

        "net_surprise_20_rank",

        "net_surprise_40_rank",

        "quality_surprise_20_rank",

        "quality_residual_rank",

        "net_surprise_20_40_combo_rank",

        "combo16_quality_residual_rank",

        "quality_residual_ewm5_rank",

        "combo16_quality20_rank",

        # new
        "factor17_ewm3_rank",

        "factor17_ewm5_rank",

        "combo17_factor18_rank",
    ]

    print("")
    print(
        "factor statistics:"
    )

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

    print("=" * 80)


# ============================================================
# Main
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
        or cfg[
            "output"
        ][
            "daily_parquet"
        ]
    )

    output_path = (
        ai_cfg.get(
            "output_parquet",
            "../data/factor/"
            "people_daily_ai_policy_factor.parquet",
        )
    )

    original_npz = (
        ai_cfg.get(
            "original_factor_npz",
            "../data/factor/"
            "people_daily_industry_factor.npz",
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