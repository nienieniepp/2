# sentiment-trend-forecasting

## 1. 项目简介
本项目面向 COMM7340，主题为“基于深度学习的商品评论情绪演化分析与预测”。

## 2. 课程要求对应说明
- Python + 深度学习 + NLP + 时序分析
- 同时提供 Notebook 与 Flask 展示

## 3. 数据集说明
- Kaggle: `snap/amazon-fine-food-reviews`
- 优先自动下载，若 `data/raw/Reviews.csv` 已存在则直接使用

## 4. 技术路线
A 数据下载与清洗；B DistilBERT 三分类；C 情绪分数生成；D 周级时序构建；E LSTM 预测；F 基线对比与可视化。

## 5. 文件结构
见仓库目录树。

## 6. 六人代码分工
- 成员1：`src/01_download_data.py`, `src/02_preprocess_data.py`
- 成员2：`src/03_train_sentiment_model.py`
- 成员3：`src/04_evaluate_sentiment_model.py`, `src/05_generate_sentiment_scores.py`
- 成员4：`src/06_build_timeseries.py`, `src/07_train_lstm_forecaster.py`
- 成员5：`src/08_baseline_models.py`, `src/09_visualize_results.py`
- 成员6：`app/*`, `src/10_run_full_pipeline.py`

## 7. 环境安装方法
```bash
pip install -r requirements.txt
```

## 8. 数据自动下载说明
```bash
python src/01_download_data.py
```

## 9. Notebook 运行方法
打开 `notebooks/01_full_pipeline_sentiment_trend_forecasting.ipynb` 按顺序运行。

## 10. 命令行脚本运行方法
按 `src/01` 到 `src/09` 顺序执行，或直接运行 `python src/10_run_full_pipeline.py`。

## 11. Flask 网页版运行方法
```bash
cd app
python app.py
```

## 12. 主要结果输出位置
`outputs/figures`, `outputs/metrics`, `outputs/predictions`, `outputs/timeseries`。

## 13. 报告和 PPT 可用内容
见 `reports/` 与 `docs/project_methodology.md`。

## 14. 常见问题
- 无 GPU：可用 CPU，但 DistilBERT 训练慢。
- 无 Kaggle 权限：先配置 Kaggle 凭证，或手动放置 `data/raw/Reviews.csv`。
- 模型不存在：先运行训练脚本。
