# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from config import FACTOR_DIR


INPUT_FILE = (
    FACTOR_DIR
    / "people_daily_industry_factor_alphasim.parquet"
)

OUTPUT_FILE = (
    FACTOR_DIR
    / "people_daily_industry_factor.npz"
)


FACTOR_COLUMNS = [
    "attention_20_rank",
    "attention_60_rank",
    "delta_attention_rank",
    "surprise_20_rank",
]


def main():

    df = pd.read_parquet(INPUT_FILE)

    df["date"] = df["date"].astype(np.int32)
    df["framework_code"] = (
        df["framework_code"]
        .astype(int)
        .astype(np.int32)
    )

    dates = np.array(
        sorted(df["date"].unique()),
        dtype=np.int32
    )

    codes = np.array(
        sorted(df["framework_code"].unique()),
        dtype=np.int32
    )

    print("dates:", len(dates))
    print("industries:", len(codes))
    print("date:", dates[0], "->", dates[-1])
    print("codes:", codes)

    data = {
        "dates": dates,
        "codes": codes,
    }

    for factor in FACTOR_COLUMNS:

        panel = (
            df.pivot(
                index="date",
                columns="framework_code",
                values=factor,
            )
            .reindex(
                index=dates,
                columns=codes,
            )
        )

        matrix = panel.to_numpy(
            dtype=np.float32
        )

        data[factor] = matrix

        print(
            factor,
            matrix.shape,
            "finite=",
            np.isfinite(matrix).sum()
        )

    np.savez_compressed(
        OUTPUT_FILE,
        **data
    )

    print()
    print("[done]", OUTPUT_FILE)


if __name__ == "__main__":
    main()