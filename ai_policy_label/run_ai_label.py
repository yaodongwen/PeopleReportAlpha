# -*- coding: utf-8 -*-

import argparse
import json
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import pandas as pd
import yaml

from providers import build_provider
from utils import (
    load_prompt,
    render_prompt,
    make_task_id,
    append_jsonl,
    load_done_ids,
    parse_json_object,
    validate_label,
)


SYSTEM_PROMPT = (
    "你是中国A股行业政策研究员。"
    "必须严格按照用户提供的定义判断指定行业的基本面影响方向，"
    "并且只输出合法JSON。"
)


STOP_REQUESTED = False


def handle_signal(signum, frame):
    global STOP_REQUESTED

    if not STOP_REQUESTED:
        print("")
        print("[stop] 收到退出信号，停止提交新任务。")
        print("[stop] 等待当前已提交任务结束后退出。")
        STOP_REQUESTED = True
    else:
        print("")
        print("[stop] 再次收到退出信号，立即退出。")
        sys.exit(1)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def load_inputs(cfg):

    inp = cfg["input"]

    article_industry_path = inp["article_industry_path"]
    news_path = inp["news_path"]

    print("[input] article_industry:", article_industry_path)
    print("[input] news:", news_path)

    ai = pd.read_parquet(
        article_industry_path
    )

    news = pd.read_parquet(
        news_path
    )

    join_key = inp["join_key"]

    required_ai = [
        join_key,
        inp["industry_col"],
    ]

    missing_ai = [
        c
        for c in required_ai
        if c not in ai.columns
    ]

    if missing_ai:
        raise KeyError(
            "article_industry missing columns: %s"
            % missing_ai
        )

    news_cols = [
        join_key,
        inp["title_col"],
        inp["text_col"],
        inp["date_col"],
    ]

    news_cols = list(
        dict.fromkeys(
            news_cols
        )
    )

    missing_news = [
        c
        for c in news_cols
        if c not in news.columns
    ]

    if missing_news:
        raise KeyError(
            "news_clean missing columns: %s"
            % missing_news
        )

    news_small = (
        news[
            news_cols
        ]
        .drop_duplicates(
            subset=[join_key],
            keep="first"
        )
    )

    df = ai.merge(
        news_small,
        how="left",
        on=join_key,
        validate="many_to_one"
    )

    return df


def build_tasks(
    df,
    cfg
):

    inp = cfg["input"]

    join_key = inp["join_key"]
    industry_col = inp["industry_col"]
    title_col = inp["title_col"]
    text_col = inp["text_col"]

    tasks = []

    records = df.to_dict(
        "records"
    )

    for row in records:

        article_id = row[
            join_key
        ]

        industry = row[
            industry_col
        ]

        task_id = make_task_id(
            article_id,
            industry
        )

        tasks.append(
            {
                "task_id":
                    task_id,

                "article_id":
                    article_id,

                "industry":
                    industry,

                "title":
                    row.get(
                        title_col,
                        ""
                    ),

                "text":
                    row.get(
                        text_col,
                        ""
                    ),
            }
        )

    limit = int(
        cfg["runtime"].get(
            "limit",
            0
        )
    )

    if limit > 0:
        tasks = tasks[
            :limit
        ]

    return tasks


def process_one(
    task,
    cfg,
    provider,
    prompt_template
):

    start_time = time.time()

    max_retries = int(
        cfg["runtime"].get(
            "max_retries",
            3
        )
    )

    max_text_chars = int(
        cfg["prompt"].get(
            "max_text_chars",
            6000
        )
    )

    user_prompt = render_prompt(
        prompt_template,
        task["industry"],
        task["title"],
        task["text"],
        max_text_chars,
    )

    last_error = None
    last_raw = None

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            raw = provider.chat(
                SYSTEM_PROMPT,
                user_prompt
            )

            last_raw = raw

            obj = (
                parse_json_object(
                    raw
                )
            )

            label = (
                validate_label(
                    obj
                )
            )

            effective_direction = (
                label[
                    "direction"
                ]
            )

            if (
                label["relevance"]
                <
                float(
                    cfg["label"][
                        "min_relevance"
                    ]
                )
            ):
                effective_direction = 0

            if (
                label["confidence"]
                <
                float(
                    cfg["label"][
                        "min_confidence"
                    ]
                )
            ):
                effective_direction = 0

            elapsed = (
                time.time()
                - start_time
            )

            return {
                "ok":
                    True,

                "task_id":
                    task[
                        "task_id"
                    ],

                "article_id":
                    task[
                        "article_id"
                    ],

                "industry":
                    task[
                        "industry"
                    ],

                "title":
                    str(
                        task.get(
                            "title",
                            ""
                        )
                    )[:120],

                "direction":
                    label[
                        "direction"
                    ],

                "effective_direction":
                    effective_direction,

                "confidence":
                    label[
                        "confidence"
                    ],

                "relevance":
                    label[
                        "relevance"
                    ],

                "policy_strength":
                    label[
                        "policy_strength"
                    ],

                "policy_type":
                    label[
                        "policy_type"
                    ],

                "reason":
                    label[
                        "reason"
                    ],

                "elapsed_seconds":
                    elapsed,

                "attempt":
                    attempt,
            }

        except Exception as e:

            last_error = repr(
                e
            )

            if (
                attempt
                < max_retries
            ):

                sleep_seconds = min(
                    2 ** (
                        attempt - 1
                    ),
                    8
                )

                time.sleep(
                    sleep_seconds
                )

    elapsed = (
        time.time()
        - start_time
    )

    return {
        "ok":
            False,

        "task_id":
            task[
                "task_id"
            ],

        "article_id":
            task[
                "article_id"
            ],

        "industry":
            task[
                "industry"
            ],

        "title":
            str(
                task.get(
                    "title",
                    ""
                )
            )[:120],

        "error":
            last_error,

        "raw":
            last_raw,

        "elapsed_seconds":
            elapsed,

        "attempt":
            max_retries,
    }


def format_seconds(
    seconds
):

    if seconds is None:
        return "-"

    seconds = max(
        0,
        int(seconds)
    )

    hours = (
        seconds // 3600
    )

    minutes = (
        seconds % 3600
    ) // 60

    secs = (
        seconds % 60
    )

    if hours > 0:
        return "%dh%02dm%02ds" % (
            hours,
            minutes,
            secs
        )

    if minutes > 0:
        return "%dm%02ds" % (
            minutes,
            secs
        )

    return "%ds" % secs


def print_progress(
    completed,
    total,
    success,
    failed,
    elapsed_total,
    recent_times,
    result
):

    if completed <= 0:
        return

    speed_per_sec = (
        completed
        / max(
            elapsed_total,
            1e-9
        )
    )

    speed_per_min = (
        speed_per_sec
        * 60.0
    )

    remaining = (
        total
        - completed
    )

    if speed_per_sec > 0:
        eta_seconds = (
            remaining
            / speed_per_sec
        )
    else:
        eta_seconds = None

    if recent_times:
        avg_recent = (
            sum(recent_times)
            / len(recent_times)
        )
    else:
        avg_recent = 0.0

    success_rate = (
        100.0
        * success
        / completed
    )

    if result["ok"]:

        print(
            "[done] %d/%d"
            " | %.2fs"
            " | avg %.2fs"
            " | %.2f/min"
            " | ETA %s"
            " | ok %.1f%%"
            " | %s"
            " | dir=%s eff=%s"
            " rel=%.2f conf=%.2f"
            % (
                completed,
                total,

                result[
                    "elapsed_seconds"
                ],

                avg_recent,

                speed_per_min,

                format_seconds(
                    eta_seconds
                ),

                success_rate,

                str(
                    result[
                        "industry"
                    ]
                ),

                result[
                    "direction"
                ],

                result[
                    "effective_direction"
                ],

                result[
                    "relevance"
                ],

                result[
                    "confidence"
                ],
            )
        )

    else:

        print(
            "[fail] %d/%d"
            " | %.2fs"
            " | %.2f/min"
            " | ETA %s"
            " | ok %.1f%%"
            " | %s"
            " | %s"
            % (
                completed,
                total,

                result[
                    "elapsed_seconds"
                ],

                speed_per_min,

                format_seconds(
                    eta_seconds
                ),

                success_rate,

                str(
                    result[
                        "industry"
                    ]
                ),

                result[
                    "error"
                ],
            )
        )


def build_final_parquet(
    df,
    cfg
):

    print("")
    print(
        "[3/4] building final parquet..."
    )

    raw_jsonl = (
        cfg["output"][
            "raw_jsonl"
        ]
    )

    labels = []

    p = Path(
        raw_jsonl
    )

    if p.exists():

        with p.open(
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = (
                    line.strip()
                )

                if not line:
                    continue

                try:

                    obj = json.loads(
                        line
                    )

                    if obj.get(
                        "ok"
                    ):
                        labels.append(
                            obj
                        )

                except Exception:
                    continue

    label_df = pd.DataFrame(
        labels
    )

    if label_df.empty:

        print(
            "[warning] no successful labels found"
        )

        return

    label_df = (
        label_df
        .drop_duplicates(
            subset=[
                "article_id",
                "industry",
            ],
            keep="last"
        )
    )

    join_key = (
        cfg["input"][
            "join_key"
        ]
    )

    industry_col = (
        cfg["input"][
            "industry_col"
        ]
    )

    final = df.merge(
        label_df,

        left_on=[
            join_key,
            industry_col
        ],

        right_on=[
            "article_id",
            "industry"
        ],

        how="left",

        suffixes=(
            "",
            "_label"
        ),

        validate="many_to_one",
    )

    output_path = (
        cfg["output"][
            "final_parquet"
        ]
    )

    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final.to_parquet(
        output_path,
        index=False
    )

    labeled_count = int(
        final[
            "effective_direction"
        ]
        .notna()
        .sum()
    )

    print(
        "final rows:",
        len(final)
    )

    print(
        "labeled:",
        labeled_count
    )

    print(
        "output:",
        output_path
    )


def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--config",
        default="config.yaml"
    )

    args = (
        parser.parse_args()
    )

    with open(
        args.config,
        "r",
        encoding="utf-8"
    ) as f:

        cfg = (
            yaml.safe_load(
                f
            )
        )

    output_cfg = (
        cfg["output"]
    )

    runtime_cfg = (
        cfg["runtime"]
    )

    Path(
        output_cfg[
            "work_dir"
        ]
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "[1/4] loading input..."
    )

    df = (
        load_inputs(
            cfg
        )
    )

    print(
        "rows:",
        len(df)
    )

    all_tasks = (
        build_tasks(
            df,
            cfg
        )
    )

    raw_jsonl = (
        output_cfg[
            "raw_jsonl"
        ]
    )

    error_jsonl = (
        output_cfg[
            "error_jsonl"
        ]
    )

    done_ids = set()

    if runtime_cfg.get(
        "resume",
        True
    ):

        done_ids = (
            load_done_ids(
                raw_jsonl
            )
        )

    tasks = [
        x
        for x in all_tasks
        if x["task_id"]
        not in done_ids
    ]

    total = len(
        tasks
    )

    print(
        "tasks to run:",
        total
    )

    print(
        "already done:",
        len(done_ids)
    )

    if total == 0:

        print(
            "[info] no tasks to run"
        )

        build_final_parquet(
            df,
            cfg
        )

        return

    provider = (
        build_provider(
            cfg
        )
    )

    prompt_template = (
        load_prompt(
            cfg[
                "prompt"
            ][
                "template_path"
            ]
        )
    )

    workers = int(
        runtime_cfg.get(
            "workers",
            1
        )
    )

    max_in_flight = int(
        runtime_cfg.get(
            "max_in_flight",
            workers * 2
        )
    )

    if max_in_flight < workers:
        max_in_flight = workers

    log_every = int(
        runtime_cfg.get(
            "log_every",
            1
        )
    )

    print("")
    print(
        "[2/4] AI labeling..."
    )

    print(
        "workers:",
        workers
    )

    print(
        "max_in_flight:",
        max_in_flight
    )

    print(
        "provider:",
        cfg["provider"]
    )

    print("")

    success = 0
    failed = 0
    completed = 0

    started_at = (
        time.time()
    )

    recent_times = deque(
        maxlen=50
    )

    task_iter = iter(
        tasks
    )

    executor = ThreadPoolExecutor(
        max_workers=workers
    )

    futures = {}

    try:

        # -----------------------------
        # 初始填充 in-flight 队列
        # -----------------------------

        while (
            len(futures)
            < max_in_flight
        ):

            try:

                task = next(
                    task_iter
                )

            except StopIteration:
                break

            future = executor.submit(
                process_one,
                task,
                cfg,
                provider,
                prompt_template
            )

            futures[
                future
            ] = task

        # -----------------------------
        # 持续消费 + 补充任务
        # -----------------------------

        while futures:

            done_set, _ = wait(
                list(
                    futures.keys()
                ),
                return_when=FIRST_COMPLETED
            )

            for future in done_set:

                task = futures.pop(
                    future
                )

                try:

                    result = (
                        future.result()
                    )

                except Exception as e:

                    result = {
                        "ok":
                            False,

                        "task_id":
                            task[
                                "task_id"
                            ],

                        "article_id":
                            task[
                                "article_id"
                            ],

                        "industry":
                            task[
                                "industry"
                            ],

                        "title":
                            str(
                                task.get(
                                    "title",
                                    ""
                                )
                            )[:120],

                        "error":
                            repr(e),

                        "raw":
                            None,

                        "elapsed_seconds":
                            0.0,

                        "attempt":
                            0,
                    }

                completed += 1

                recent_times.append(
                    float(
                        result.get(
                            "elapsed_seconds",
                            0.0
                        )
                    )
                )

                if result[
                    "ok"
                ]:

                    append_jsonl(
                        raw_jsonl,
                        result
                    )

                    success += 1

                else:

                    append_jsonl(
                        error_jsonl,
                        result
                    )

                    failed += 1

                elapsed_total = (
                    time.time()
                    - started_at
                )

                if (
                    completed
                    % log_every
                    == 0
                    or completed
                    == total
                ):

                    print_progress(
                        completed,
                        total,
                        success,
                        failed,
                        elapsed_total,
                        recent_times,
                        result
                    )

                # ---------------------
                # 补一个新任务
                # ---------------------

                if not STOP_REQUESTED:

                    try:

                        next_task = next(
                            task_iter
                        )

                    except StopIteration:
                        next_task = None

                    if next_task is not None:

                        new_future = (
                            executor.submit(
                                process_one,
                                next_task,
                                cfg,
                                provider,
                                prompt_template
                            )
                        )

                        futures[
                            new_future
                        ] = next_task

            if STOP_REQUESTED:

                print("")
                print(
                    "[stop] 不再提交新任务，"
                    "等待当前 in-flight 任务完成..."
                )

                # 清空尚未消费的 task_iter
                task_iter = iter(
                    []
                )

    finally:

        executor.shutdown(
            wait=True
        )

    elapsed_total = (
        time.time()
        - started_at
    )

    print("")
    print("=" * 72)

    print(
        "AI LABEL SUMMARY"
    )

    print("=" * 72)

    print(
        "completed:",
        completed
    )

    print(
        "success:",
        success
    )

    print(
        "failed:",
        failed
    )

    if completed > 0:

        print(
            "success rate:",
            "%.2f%%"
            % (
                100.0
                * success
                / completed
            )
        )

        print(
            "avg wall time:",
            "%.2fs/item"
            % (
                elapsed_total
                / completed
            )
        )

        print(
            "throughput:",
            "%.2f items/min"
            % (
                completed
                / max(
                    elapsed_total,
                    1e-9
                )
                * 60.0
            )
        )

    print(
        "elapsed:",
        format_seconds(
            elapsed_total
        )
    )

    print("=" * 72)

    # -----------------------------
    # 构建最终 parquet
    # -----------------------------

    build_final_parquet(
        df,
        cfg
    )

    print("")
    print(
        "[4/4] done"
    )


if __name__ == "__main__":
    main()