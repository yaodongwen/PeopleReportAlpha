# -*- coding: utf-8 -*-

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    load_prompt,
    render_prompt,
    make_task_id,
    append_jsonl,
    load_done_ids,
    parse_json_object,
    validate_label,
)


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = (
    "你是中国A股行业政策研究员。"
    "必须严格按照用户提供的定义判断指定行业的基本面影响方向，"
    "并且只输出合法JSON。"
)


STOP_REQUESTED = False


# ============================================================
# Signal handling
# ============================================================

def handle_signal(signum, frame):
    global STOP_REQUESTED

    if not STOP_REQUESTED:
        print("")
        print("[stop] 收到退出信号。")
        print("[stop] 当前 batch 完成后停止，不再继续提交新 batch。")
        STOP_REQUESTED = True
    else:
        print("")
        print("[stop] 再次收到退出信号，立即退出。")
        sys.exit(1)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ============================================================
# Input
# ============================================================

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


# ============================================================
# Build tasks
# ============================================================

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


# ============================================================
# Device / model
# ============================================================

def get_torch_dtype(
    dtype_name
):

    dtype_name = str(
        dtype_name
    ).lower()

    if dtype_name in (
        "bf16",
        "bfloat16",
    ):
        return torch.bfloat16

    if dtype_name in (
        "fp16",
        "float16",
    ):
        return torch.float16

    if dtype_name in (
        "fp32",
        "float32",
    ):
        return torch.float32

    raise ValueError(
        "unsupported dtype: %s"
        % dtype_name
    )


def load_model_and_tokenizer(
    cfg
):

    model_cfg = cfg["gpu_model"]

    model_name = model_cfg[
        "model"
    ]

    dtype_name = model_cfg.get(
        "dtype",
        "bfloat16"
    )

    torch_dtype = (
        get_torch_dtype(
            dtype_name
        )
    )

    print("")
    print("[model] loading:", model_name)
    print("[model] dtype:", dtype_name)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
    )

    # Decoder-only 模型 batch generate 推荐 left padding
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="cuda",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    )

    model.eval()

    print(
        "[model] device:",
        next(
            model.parameters()
        ).device
    )

    print(
        "[model] CUDA:",
        torch.cuda.get_device_name(0)
    )

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / 1024 ** 3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024 ** 3
        )

        print(
            "[model] GPU allocated: %.2f GB"
            % allocated
        )

        print(
            "[model] GPU reserved: %.2f GB"
            % reserved
        )

    return (
        model,
        tokenizer
    )


# ============================================================
# Prompt
# ============================================================

def build_chat_text(
    task,
    cfg,
    tokenizer,
    prompt_template
):

    max_text_chars = int(
        cfg["prompt"].get(
            "max_text_chars",
            2000
        )
    )

    user_prompt = render_prompt(
        prompt_template,
        task["industry"],
        task["title"],
        task["text"],
        max_text_chars,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    chat_text = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,

            # Qwen3：硬关闭 thinking
            enable_thinking=False,
        )
    )

    return chat_text


# ============================================================
# Batch inference
# ============================================================

def run_batch(
    batch_tasks,
    cfg,
    model,
    tokenizer,
    prompt_template
):

    runtime = cfg["runtime"]

    max_input_tokens = int(
        runtime.get(
            "max_input_tokens",
            2048
        )
    )

    max_new_tokens = int(
        runtime.get(
            "max_tokens",
            150
        )
    )

    temperature = float(
        runtime.get(
            "temperature",
            0.0
        )
    )

    texts = []

    for task in batch_tasks:

        text = build_chat_text(
            task,
            cfg,
            tokenizer,
            prompt_template
        )

        texts.append(
            text
        )

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )

    encoded = {
        k: v.to("cuda")
        for k, v
        in encoded.items()
    }

    input_len = encoded[
        "input_ids"
    ].shape[1]

    generation_kwargs = {
        "max_new_tokens":
            max_new_tokens,

        "pad_token_id":
            tokenizer.pad_token_id,

        "eos_token_id":
            tokenizer.eos_token_id,

        "use_cache":
            True,
    }

    if temperature <= 0:

        generation_kwargs[
            "do_sample"
        ] = False

    else:

        generation_kwargs[
            "do_sample"
        ] = True

        generation_kwargs[
            "temperature"
        ] = temperature

        top_p = runtime.get(
            "top_p",
            None
        )

        if top_p is not None:

            generation_kwargs[
                "top_p"
            ] = float(
                top_p
            )

    batch_start = time.time()

    with torch.inference_mode():

        output_ids = (
            model.generate(
                **encoded,
                **generation_kwargs
            )
        )

    torch.cuda.synchronize()

    batch_elapsed = (
        time.time()
        - batch_start
    )

    generated_ids = (
        output_ids[
            :,
            input_len:
        ]
    )

    outputs = (
        tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
    )

    results = []

    for task, raw in zip(
        batch_tasks,
        outputs
    ):

        try:

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

            results.append(
                {
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

                    "raw":
                        raw,

                    "batch_elapsed_seconds":
                        batch_elapsed,
                }
            )

        except Exception as e:

            results.append(
                {
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
                        raw,

                    "batch_elapsed_seconds":
                        batch_elapsed,
                }
            )

    return (
        results,
        batch_elapsed,
        input_len,
    )


# ============================================================
# Retry failed items one-by-one
# ============================================================

def retry_failed_single(
    failed_results,
    task_map,
    cfg,
    model,
    tokenizer,
    prompt_template
):

    retry_results = []

    for old_result in failed_results:

        task_id = old_result[
            "task_id"
        ]

        task = task_map[
            task_id
        ]

        try:

            results, elapsed, input_len = (
                run_batch(
                    [task],
                    cfg,
                    model,
                    tokenizer,
                    prompt_template
                )
            )

            retry_results.extend(
                results
            )

        except Exception as e:

            retry_results.append(
                {
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
                }
            )

    return retry_results


# ============================================================
# Progress
# ============================================================

def format_seconds(
    seconds
):

    if seconds is None:
        return "-"

    seconds = int(
        max(
            0,
            seconds
        )
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
    batch_size,
    batch_elapsed,
    input_len,
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

    success_rate = (
        success
        / completed
        * 100.0
    )

    gpu_allocated = (
        torch.cuda.memory_allocated()
        / 1024 ** 3
    )

    gpu_reserved = (
        torch.cuda.memory_reserved()
        / 1024 ** 3
    )

    print(
        "[batch] %d/%d"
        " | bs=%d"
        " | %.2fs"
        " | %.2f items/min"
        " | ETA %s"
        " | ok %.1f%%"
        " | input_tokens=%d"
        " | GPU %.1f/%.1f GB"
        % (
            completed,
            total,

            batch_size,

            batch_elapsed,

            speed_per_min,

            format_seconds(
                eta_seconds
            ),

            success_rate,

            input_len,

            gpu_allocated,
            gpu_reserved,
        )
    )


# ============================================================
# Final parquet
# ============================================================

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

    label_df = (
        pd.DataFrame(
            labels
        )
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


# ============================================================
# Main
# ============================================================

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

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available"
        )

    print("=" * 72)
    print("GPU AI LABELER")
    print("=" * 72)

    print(
        "torch:",
        torch.__version__
    )

    print(
        "torch cuda:",
        torch.version.cuda
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print("")

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

    model, tokenizer = (
        load_model_and_tokenizer(
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

    batch_size = int(
        runtime_cfg.get(
            "batch_size",
            16
        )
    )

    retry_failed = bool(
        runtime_cfg.get(
            "retry_failed",
            True
        )
    )

    print("")
    print(
        "[2/4] GPU batch labeling..."
    )

    print(
        "batch_size:",
        batch_size
    )

    print(
        "max_input_tokens:",
        runtime_cfg.get(
            "max_input_tokens",
            2048
        )
    )

    print(
        "max_tokens:",
        runtime_cfg.get(
            "max_tokens",
            150
        )
    )

    print("")

    success = 0
    failed = 0
    completed = 0

    started_at = (
        time.time()
    )

    task_map = {
        x["task_id"]: x
        for x in tasks
    }

    num_batches = math.ceil(
        total
        / batch_size
    )

    for batch_idx in range(
        num_batches
    ):

        if STOP_REQUESTED:
            break

        start = (
            batch_idx
            * batch_size
        )

        end = min(
            start
            + batch_size,
            total
        )

        batch_tasks = (
            tasks[
                start:end
            ]
        )

        try:

            results, batch_elapsed, input_len = (
                run_batch(
                    batch_tasks,
                    cfg,
                    model,
                    tokenizer,
                    prompt_template
                )
            )

        except torch.cuda.OutOfMemoryError:

            torch.cuda.empty_cache()

            print("")
            print(
                "[OOM] batch_size=%d too large"
                % len(
                    batch_tasks
                )
            )

            print(
                "[OOM] reduce runtime.batch_size"
            )

            raise

        except Exception as e:

            print("")
            print(
                "[batch error]",
                repr(e)
            )

            results = []

            for task in batch_tasks:

                results.append(
                    {
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
                    }
                )

            batch_elapsed = 0.0
            input_len = 0

        failed_this_batch = []

        for result in results:

            if result[
                "ok"
            ]:

                append_jsonl(
                    raw_jsonl,
                    result
                )

                success += 1

            else:

                failed_this_batch.append(
                    result
                )

        # -------------------------------------
        # 对 batch 中解析失败的样本单条重试
        # -------------------------------------

        if (
            retry_failed
            and failed_this_batch
        ):

            retry_results = (
                retry_failed_single(
                    failed_this_batch,
                    task_map,
                    cfg,
                    model,
                    tokenizer,
                    prompt_template
                )
            )

            for result in retry_results:

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

        else:

            for result in failed_this_batch:

                append_jsonl(
                    error_jsonl,
                    result
                )

                failed += 1

        completed += len(
            batch_tasks
        )

        elapsed_total = (
            time.time()
            - started_at
        )

        print_progress(
            completed,
            total,
            success,
            failed,
            elapsed_total,
            len(
                batch_tasks
            ),
            batch_elapsed,
            input_len,
        )

    elapsed_total = (
        time.time()
        - started_at
    )

    print("")
    print("=" * 72)
    print("AI LABEL SUMMARY")
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
                success
                / completed
                * 100.0
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

    print(
        "GPU max allocated:",
        "%.2f GB"
        % (
            torch.cuda.max_memory_allocated()
            / 1024 ** 3
        )
    )

    print("=" * 72)

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