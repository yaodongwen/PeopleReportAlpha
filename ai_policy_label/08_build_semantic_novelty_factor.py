#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
from collections import deque

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer


ARTICLE_CANDIDATES = [
    "ai_policy_label/outputs/article_industry_policy.parquet",
    "outputs/article_industry_policy.parquet",
]
SIGNED_CANDIDATES = [
    "ai_policy_label/outputs/industry_signed_daily.parquet",
    "outputs/industry_signed_daily.parquet",
]
BASE_CANDIDATES = [
    "data/factor/people_daily_ai_policy_factor.npz",
    "../data/factor/people_daily_ai_policy_factor.npz",
]
OUTPUT_CANDIDATES = [
    "data/factor/people_daily_semantic_novelty_factor.npz",
    "../data/factor/people_daily_semantic_novelty_factor.npz",
]

LOOKBACK_DAYS = 730
FIRST_SIM_THRESHOLD = 0.25
ROLL_WINDOW = 20
HIST_WINDOW = 365
MIN_HISTORY = 120
STD_FLOOR = 1e-6
EPS = 1e-12

N_FEATURES = 2 ** 15
MAX_PRIOR_DOCS = 2000
PROGRESS_EVERY = 1000


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def choose_output(paths):
    for p in paths:
        parent = os.path.dirname(p)
        if not parent or os.path.isdir(parent):
            return p
    return paths[0]


def parquet_columns(path):
    """
    Read parquet column names WITHOUT pyarrow.

    This project environment already uses pandas.read_parquet successfully,
    so we use fastparquet metadata directly.
    """
    try:
        from fastparquet import ParquetFile
    except ImportError as e:
        raise ImportError(
            "fastparquet is required for this v3 script because pyarrow is not installed. "
            "Check with: python -c \"import fastparquet; print(fastparquet.__version__)\""
        ) from e

    pf = ParquetFile(path)

    # fastparquet exposes physical columns via .columns
    return list(pf.columns)


def detect(names, candidates, required=True):
    s = set(names)
    for c in candidates:
        if c in s:
            return c
    if required:
        raise KeyError("missing %s; actual=%s" % (candidates, sorted(s)))
    return None


def normalize_date(s):
    numeric = pd.to_numeric(s, errors="coerce")
    good = numeric.notna()

    if good.any():
        vals = numeric.loc[good].astype(np.int64)
        if ((vals >= 19000101) & (vals <= 21001231)).all():
            out = pd.Series(np.nan, index=s.index, dtype=float)
            out.loc[good] = vals.values
            return out

    p = pd.to_datetime(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    ok = p.notna()
    out.loc[ok] = p.loc[ok].dt.strftime("%Y%m%d").astype(np.int64)
    return out


def cs_rank_matrix(x):
    return (
        pd.DataFrame(x)
        .rank(axis=1, pct=True, method="average", na_option="keep")
        .mul(2.0)
        .sub(1.0)
        .values.astype(np.float64)
    )


def historical_surprise_rank(raw):
    z = np.full_like(raw, np.nan, dtype=np.float64)

    for j in range(raw.shape[1]):
        s = pd.Series(raw[:, j], dtype=float)
        h = s.shift(1)

        mean = h.rolling(
            HIST_WINDOW,
            min_periods=MIN_HISTORY
        ).mean()

        std = h.rolling(
            HIST_WINDOW,
            min_periods=MIN_HISTORY
        ).std()

        std = std.where(std > STD_FLOOR, np.nan)
        z[:, j] = ((s - mean) / std).values

    return z, cs_rank_matrix(z)


def load_article_minimal(path):
    names = parquet_columns(path)

    col = {
        "date": detect(names, ["date_x", "date", "date_y"]),
        "code": detect(names, ["framework_code", "index_code"]),
        "article_id": detect(names, ["article_id"], required=False),
        "article_weight": detect(names, ["article_weight"]),
        "confidence": detect(names, ["confidence"]),
        "relevance": detect(names, ["relevance"]),
        "strength": detect(names, ["policy_strength"]),
        "ok": detect(names, ["ok"], required=False),
        "title": detect(names, ["title", "title_x", "title_y"], required=False),
        "reason": detect(names, ["reason_label"], required=False),
        "context": detect(names, ["context_terms"], required=False),
        "text": detect(names, ["text"], required=False),
    }

    cols = [
        col["date"],
        col["code"],
        col["article_weight"],
        col["confidence"],
        col["relevance"],
        col["strength"],
    ]

    for k in ["article_id", "ok", "title", "reason", "context", "text"]:
        if col[k] is not None:
            cols.append(col[k])

    cols = list(dict.fromkeys(cols))

    print("reading only columns:", cols, flush=True)

    df = pd.read_parquet(
        path,
        columns=cols,
        engine="fastparquet",
    )

    return df, col


def build_semantic_text(df, col):
    parts = []

    if col["title"] is not None:
        title = (
            df[col["title"]]
            .fillna("")
            .astype(str)
            .str.slice(0, 120)
        )
        parts += [title, title]

    if col["reason"] is not None:
        parts.append(
            df[col["reason"]]
            .fillna("")
            .astype(str)
            .str.slice(0, 200)
        )

    if col["context"] is not None:
        parts.append(
            df[col["context"]]
            .fillna("")
            .astype(str)
            .str.slice(0, 300)
        )

    if col["text"] is not None:
        parts.append(
            df[col["text"]]
            .fillna("")
            .astype(str)
            .str.slice(0, 300)
        )

    if not parts:
        raise RuntimeError("no usable text columns")

    out = parts[0].copy()
    for p in parts[1:]:
        out = out + " " + p

    return (
        out
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def score_industry_fast(g, vectorizer, code):
    g = (
        g.sort_values(["date_dt", "_row_id"])
        .reset_index(drop=True)
        .copy()
    )

    print(
        "  vectorizing code=%d rows=%d" % (int(code), len(g)),
        flush=True
    )

    X = vectorizer.transform(
        g["semantic_text"].tolist()
    ).tocsr()

    dates = (
        g["date_dt"]
        .values
        .astype("datetime64[D]")
    )

    max_sim = np.zeros(len(g), dtype=np.float32)

    unique_dates, starts = np.unique(
        dates,
        return_index=True
    )
    ends = np.r_[starts[1:], len(g)]

    history_idx = deque()
    processed = 0
    t0 = time.time()

    for day, start, end in zip(unique_dates, starts, ends):
        lower = day - np.timedelta64(LOOKBACK_DAYS, "D")

        while history_idx and dates[history_idx[0]] < lower:
            history_idx.popleft()

        while len(history_idx) > MAX_PRIOR_DOCS:
            history_idx.popleft()

        if history_idx:
            hist_arr = np.fromiter(history_idx, dtype=np.int64)
            X_prior = X[hist_arr]

            for i in range(start, end):
                sims = X[i] @ X_prior.T

                if sparse.issparse(sims):
                    max_sim[i] = (
                        float(sims.data.max())
                        if sims.nnz
                        else 0.0
                    )
                else:
                    max_sim[i] = float(np.max(sims))

                processed += 1

                if processed % PROGRESS_EVERY == 0:
                    print(
                        "    code=%d processed=%d/%d history=%d elapsed=%.1fs"
                        % (
                            int(code),
                            processed,
                            len(g),
                            len(history_idx),
                            time.time() - t0,
                        ),
                        flush=True,
                    )

        else:
            processed += end - start

        # 同日文章计算完再加入历史，避免同日互相“泄漏”
        for i in range(start, end):
            history_idx.append(i)

        while len(history_idx) > MAX_PRIOR_DOCS:
            history_idx.popleft()

    max_sim = np.clip(max_sim, 0.0, 1.0)

    g["max_prior_similarity_730"] = max_sim.astype(np.float64)
    g["semantic_novelty"] = 1.0 - g["max_prior_similarity_730"]
    g["semantic_first_like_730"] = (
        g["max_prior_similarity_730"] < FIRST_SIM_THRESHOLD
    )

    print(
        "  finished code=%d elapsed=%.1fs"
        % (int(code), time.time() - t0),
        flush=True,
    )

    return g


def main():
    total_t0 = time.time()

    article_path = first_existing(ARTICLE_CANDIDATES)
    signed_path = first_existing(SIGNED_CANDIDATES)
    base_path = first_existing(BASE_CANDIDATES)

    if not article_path or not signed_path or not base_path:
        raise FileNotFoundError("required input missing")

    output_path = choose_output(OUTPUT_CANDIDATES)

    print("=" * 78, flush=True)
    print("BUILD SEMANTIC FIRST-APPEARANCE NOVELTY V3 (NO PYARROW)", flush=True)
    print("=" * 78, flush=True)
    print("article:", article_path, flush=True)
    print("signed :", signed_path, flush=True)
    print("base   :", base_path, flush=True)
    print("lookback:", LOOKBACK_DAYS, flush=True)
    print("threshold:", FIRST_SIM_THRESHOLD, flush=True)
    print("max prior docs:", MAX_PRIOR_DOCS, flush=True)
    print("hash features:", N_FEATURES, flush=True)

    with np.load(base_path) as z:
        dates = z["dates"].astype(np.int64)
        codes = z["codes"].astype(np.int64)

    valid_dates = set(dates.tolist())
    valid_codes = set(codes.tolist())

    t0 = time.time()
    art, col = load_article_minimal(article_path)

    print(
        "article parquet loaded: %d rows, %.1fs"
        % (len(art), time.time() - t0),
        flush=True,
    )

    art["_row_id"] = np.arange(len(art), dtype=np.int64)
    art["date_int"] = normalize_date(art[col["date"]])
    art["code"] = pd.to_numeric(art[col["code"]], errors="coerce")

    if col["ok"] is not None:
        if art[col["ok"]].dtype == bool:
            art = art[art[col["ok"]]].copy()
        else:
            ok = (
                art[col["ok"]]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "1.0"])
            )
            art = art[ok].copy()

    for c in [
        col["article_weight"],
        col["confidence"],
        col["relevance"],
        col["strength"],
    ]:
        art[c] = pd.to_numeric(art[c], errors="coerce")

    art = art[
        art["date_int"].notna()
        & art["code"].notna()
    ].copy()

    art["date_int"] = art["date_int"].astype(np.int64)
    art["code"] = art["code"].astype(np.int64)

    art = art[
        art["date_int"].isin(valid_dates)
        & art["code"].isin(valid_codes)
    ].copy()

    art["date_dt"] = pd.to_datetime(
        art["date_int"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    art["ai_weight"] = (
        art[col["article_weight"]].fillna(0.0)
        * art[col["confidence"]].fillna(0.0)
        * art[col["relevance"]].fillna(0.0)
        * art[col["strength"]].fillna(0.0)
    )

    print("building semantic_text...", flush=True)

    art["semantic_text"] = build_semantic_text(art, col)
    art = art[art["semantic_text"].str.len() > 0].copy()

    print(
        "usable article-industry rows:",
        len(art),
        flush=True,
    )

    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        n_features=N_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )

    parts = []
    groups = list(art.groupby("code", sort=True))

    for idx, (code, g) in enumerate(groups, 1):
        print(
            "\n[%02d/%02d] code=%d rows=%d"
            % (idx, len(groups), int(code), len(g)),
            flush=True,
        )

        parts.append(
            score_industry_fast(
                g,
                vectorizer,
                code,
            )
        )

    scored = pd.concat(parts, ignore_index=True)

    scored["first_like_weight"] = (
        scored["ai_weight"]
        * scored["semantic_first_like_730"].astype(np.float64)
    )

    scored["semantic_novelty_weight"] = (
        scored["ai_weight"]
        * scored["semantic_novelty"]
    )

    novelty_daily = (
        scored.groupby(["date_int", "code"], as_index=False)
        .agg(
            first_like_weight=("first_like_weight", "sum"),
            semantic_novelty_weight=("semantic_novelty_weight", "sum"),
        )
    )

    print("loading signed daily...", flush=True)

    signed = pd.read_parquet(
        signed_path,
        columns=[
            "date",
            "industry",
            "quality_attention",
            "total_articles",
        ],
        engine="fastparquet",
    )

    signed["date_int"] = normalize_date(signed["date"])
    signed["code"] = pd.to_numeric(
        signed["industry"],
        errors="coerce"
    )
    signed["quality_attention"] = pd.to_numeric(
        signed["quality_attention"],
        errors="coerce"
    ).fillna(0.0)
    signed["total_articles"] = pd.to_numeric(
        signed["total_articles"],
        errors="coerce"
    ).fillna(0.0)

    signed = signed[
        signed["date_int"].notna()
        & signed["code"].notna()
    ].copy()

    signed["date_int"] = signed["date_int"].astype(np.int64)
    signed["code"] = signed["code"].astype(np.int64)

    signed = signed[
        signed["date_int"].isin(valid_dates)
        & signed["code"].isin(valid_codes)
    ]

    daily = (
        signed.groupby(["date_int", "code"], as_index=False)
        .agg(
            quality_attention=("quality_attention", "sum"),
            total_articles=("total_articles", "max"),
        )
    )

    full = pd.MultiIndex.from_product(
        [dates, codes],
        names=["date_int", "code"]
    )

    dg = (
        daily
        .set_index(["date_int", "code"])
        .reindex(full)
        .reset_index()
    )

    ng = (
        novelty_daily
        .set_index(["date_int", "code"])
        .reindex(full)
        .fillna(0.0)
        .reset_index()
    )

    dg["quality_attention"] = dg["quality_attention"].fillna(0.0)
    dg["total_articles"] = dg["total_articles"].fillna(0.0)

    shape = (len(dates), len(codes))

    quality = (
        dg["quality_attention"]
        .values
        .reshape(shape)
        .astype(np.float64)
    )

    denom = (
        dg["total_articles"]
        .values
        .reshape(shape)
        .astype(np.float64)
    )

    first_weight = (
        ng["first_like_weight"]
        .values
        .reshape(shape)
        .astype(np.float64)
    )

    novelty_weight = (
        ng["semantic_novelty_weight"]
        .values
        .reshape(shape)
        .astype(np.float64)
    )

    first_attention = np.divide(
        first_weight,
        denom,
        out=np.zeros_like(first_weight),
        where=denom > 0,
    )

    novelty_attention = np.divide(
        novelty_weight,
        denom,
        out=np.zeros_like(novelty_weight),
        where=denom > 0,
    )

    quality20 = (
        pd.DataFrame(quality)
        .rolling(ROLL_WINDOW, min_periods=1)
        .sum()
    )

    first20 = (
        pd.DataFrame(first_attention)
        .rolling(ROLL_WINDOW, min_periods=1)
        .sum()
    )

    novelty20 = (
        pd.DataFrame(novelty_attention)
        .rolling(ROLL_WINDOW, min_periods=1)
        .sum()
    )

    first_share20 = (
        first20
        / quality20.where(quality20 > EPS, np.nan)
    ).values.astype(np.float64)

    novelty_share20 = (
        novelty20
        / quality20.where(quality20 > EPS, np.nan)
    ).values.astype(np.float64)

    print("building historical surprise...", flush=True)

    first_z, first_rank = historical_surprise_rank(first_share20)
    novelty_z, novelty_rank = historical_surprise_rank(novelty_share20)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        dates=dates,
        codes=codes,
        semantic_first_appearance_surprise_20_rank=first_rank,
        semantic_novelty_surprise_20_rank=novelty_rank,
        semantic_first_share_20=first_share20,
        semantic_novelty_share_20=novelty_share20,
        semantic_first_surprise_20=first_z,
        semantic_novelty_surprise_20=novelty_z,
    )

    sims = scored["max_prior_similarity_730"].values.astype(np.float64)
    qs = np.quantile(sims, [0.1, 0.25, 0.5, 0.75, 0.9])

    print("", flush=True)
    print("semantic diagnostics:", flush=True)
    print("  scored rows:", len(scored), flush=True)
    print(
        "  first-like ratio: %.4f"
        % scored["semantic_first_like_730"].mean(),
        flush=True,
    )
    print(
        "  maxsim q10/q25/q50/q75/q90:",
        " ".join("%.3f" % x for x in qs),
        flush=True,
    )
    print(
        "  semantic first rank finite: %.4f"
        % np.isfinite(first_rank).mean(),
        flush=True,
    )
    print(
        "  semantic novelty rank finite: %.4f"
        % np.isfinite(novelty_rank).mean(),
        flush=True,
    )
    print("", flush=True)
    print("saved:", output_path, flush=True)
    print(
        "total elapsed: %.1fs"
        % (time.time() - total_t0),
        flush=True,
    )
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
