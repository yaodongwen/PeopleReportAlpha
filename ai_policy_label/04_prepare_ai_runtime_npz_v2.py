# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


MATRIX_COLS = [

    # ======================================================
    # Raw daily signals
    # ======================================================

    "positive_attention",
    "negative_attention",
    "net_attention",

    "quality_attention",
    "high_conf_negative_attention",
    "constraint_attention",

    # ======================================================
    # Rolling
    # ======================================================

    "positive_20",
    "negative_20",

    "net_10",
    "net_20",
    "net_30",
    "net_40",

    "quality_20",
    "quality_30",
    "quality_40",

    # ======================================================
    # Surprise raw
    # ======================================================

    "positive_surprise_20",
    "negative_surprise_20",

    "net_surprise_10",
    "net_surprise_20",
    "net_surprise_30",
    "net_surprise_40",

    "quality_surprise_20",
    "quality_surprise_30",
    "quality_surprise_40",

    # ======================================================
    # factorType 1-11
    # ======================================================

    "positive_surprise_20_rank",
    "negative_surprise_20_rank",

    "net_surprise_20_rank",

    "net_attention_20_rank",

    "net_surprise_10_rank",

    "net_surprise_40_rank",

    "net_ewm10_surprise_rank",

    "quality_surprise_20_rank",

    "ai_residual_rank",

    "high_conf_negative_20_rank",

    "constraint_surprise_20_rank",

    # ======================================================
    # factorType 12-16
    # ======================================================

    # 12
    "direction_residual_rank",

    # 13
    "quality_residual_rank",

    # 14
    "quality_surprise_30_rank",

    # 15
    "quality_surprise_40_rank",

    # 16
    "net_surprise_20_40_combo_rank",
    # ======================================================
    # factorType 17-19
    # ======================================================

    # 17
    "combo16_quality_residual_rank",

    # 18
    "quality_residual_ewm5_rank",

    # 19
    "combo16_quality20_rank",
]


def make_matrix(
    df,
    value_col,
    dates,
    codes,
):

    pivot = (
        df.pivot(
            index="date_int",
            columns="industry",
            values=value_col,
        )
        .reindex(
            index=dates,
            columns=codes,
        )
    )

    return (
        pivot.to_numpy(
            dtype=np.float32
        )
    )


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

    input_path = ai_cfg.get(
        "output_parquet",
        "../data/factor/"
        "people_daily_ai_policy_factor.parquet",
    )

    output_path = ai_cfg.get(
        "output_npz",
        "../data/factor/"
        "people_daily_ai_policy_factor.npz",
    )

    print(
        "input:",
        input_path
    )

    print(
        "output:",
        output_path
    )

    df = pd.read_parquet(
        input_path
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

    if (
        "date_int"
        not in df.columns
    ):

        df["date"] = (
            pd.to_datetime(
                df["date"]
            )
        )

        df["date_int"] = (
            df["date"]
            .dt.strftime(
                "%Y%m%d"
            )
            .astype(np.int64)
        )

    # ======================================================
    # Check
    # ======================================================

    missing_cols = [
        c
        for c in MATRIX_COLS
        if c not in df.columns
    ]

    if missing_cols:

        print("")
        print(
            "Available columns:"
        )

        print(
            df.columns.tolist()
        )

        raise KeyError(
            "missing factor columns: %s"
            % missing_cols
        )

    dates = np.array(
        sorted(
            df[
                "date_int"
            ].unique()
        ),
        dtype=np.int64,
    )

    codes = np.array(
        sorted(
            df[
                "industry"
            ].unique()
        ),
        dtype="U6",
    )

    print(
        "dates:",
        len(dates)
    )

    print(
        "date range:",
        dates[0],
        "->",
        dates[-1]
    )

    print(
        "industry codes:",
        len(codes)
    )

    print(
        codes.tolist()
    )

    arrays = {

        "dates":
            dates,

        "codes":
            codes,
    }

    # ======================================================
    # Matrices
    # ======================================================

    for col in MATRIX_COLS:

        matrix = make_matrix(
            df,
            col,
            dates,
            codes,
        )

        arrays[
            col
        ] = matrix

        finite = (
            np.isfinite(
                matrix
            )
        )

        print("")
        print(
            "%s:" % col
        )

        print(
            "  shape:",
            matrix.shape
        )

        print(
            "  finite:",
            int(
                finite.sum()
            ),
            "/",
            matrix.size
        )

        if finite.any():

            print(
                "  min/max:",
                float(
                    np.nanmin(
                        matrix
                    )
                ),
                float(
                    np.nanmax(
                        matrix
                    )
                )
            )

    # ======================================================
    # Save
    # ======================================================

    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        **arrays
    )

    print("")
    print("=" * 72)

    print(
        "saved:",
        output_path
    )

    print(
        "file size: %.2f MB"
        % (
            Path(
                output_path
            ).stat().st_size
            / 1024
            / 1024
        )
    )

    print("=" * 72)


if __name__ == "__main__":
    main()