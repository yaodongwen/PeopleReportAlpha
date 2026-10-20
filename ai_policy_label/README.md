# AI Policy Label

用于对 `article_industry.parquet` 中的“文章 × 行业”样本进行本地 LLM 政策方向标注。

目录：

```text
ai_policy_label/
├── config.yaml
├── prompts/policy_direction.txt
├── providers.py
├── utils.py
├── run_ai_label.py
├── build_signed_daily.py
├── requirements.txt
└── outputs/
```

## 1. 安装

```bash
cd ai_policy_label
pip install -r requirements.txt
```

## 2. Ollama

```bash
ollama pull qwen2.5:14b
ollama serve
```

`config.yaml` 中：

```yaml
provider: ollama
```

第一次建议：

```yaml
runtime:
  limit: 300
  workers: 1
```

然后：

```bash
python run_ai_label.py --config config.yaml
```

确认标注质量后改为：

```yaml
runtime:
  limit: 0
  workers: 2
```

再次运行即可全量；成功 task_id 会自动跳过。

## 3. vLLM

启动示例：

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct --host 0.0.0.0 --port 8000
```

然后把：

```yaml
provider: vllm
```

并运行：

```bash
python run_ai_label.py --config config.yaml
```

## 4. 输出

- `outputs/policy_labels.jsonl`：成功标签，支持断点续跑
- `outputs/policy_label_errors.jsonl`：失败请求
- `outputs/article_industry_policy.parquet`：最终文章×行业标签表

## 5. 聚合正负行业日度信号

```bash
python build_signed_daily.py --config config.yaml
```

得到：

```text
outputs/industry_signed_daily.parquet
```

核心字段：

- `positive_attention`
- `negative_attention`
- `net_attention`

之后可继续沿用你现有的 10/20/40/60 日滚动、180/365/730 日历史 surprise、行业截面 rank 和 AlphaSim 回测框架。
