# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from config import FACTOR_DIR


INPUT_FILE = (
    FACTOR_DIR
    / "industry_policy_factor.parquet"
)

OUTPUT_FILE = (
    FACTOR_DIR
    / "people_daily_attention_runtime.npz"
)


def main():

    df = pd.read_parquet(INPUT_FILE)

    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.strftime("%Y%m%d")
        .astype(np.int32)
    )

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

    panel = (
        df.pivot(
            index="date",
            columns="framework_code",
            values="attention"
        )
        .reindex(
            index=dates,
            columns=codes
        )
        .fillna(0.0)
    )

    attention = panel.to_numpy(
        dtype=np.float32
    )

    np.savez_compressed(
        OUTPUT_FILE,
        dates=dates,
        codes=codes,
        attention=attention
    )

    print("dates:", len(dates))
    print("date:", dates[0], "->", dates[-1])
    print("industries:", len(codes))
    print("attention shape:", attention.shape)
    print("codes:", codes)
    print()
    print("[done]", OUTPUT_FILE)


if __name__ == "__main__":
    main()
    