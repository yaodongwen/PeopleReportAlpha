# PeopleDailyFactor

人民日报新闻 -> SW1 行业日频因子。

## 流程
1. 把数据从https://huggingface.co/datasets/Papersnake/people_daily_news/tree/main下载后，放到data/raw中，然后运行`python 01_prepare_news.py`，清洗原始新闻。然后确认：news.parquet生成。
2. `python 02_classify_sw1.py`运行，新闻分类到 SW1 行业
3. `python 03_build_daily_signal.py`，构建日频行业信号，得到data/factor/industry_daily.parquet

## 目录
- `data/raw/`       原始数据
- `data/processed/` 清洗 / 分类结果
- `data/factor/`    最终因子
