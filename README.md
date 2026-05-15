# sentiment-trend-forecasting

## 1. 项目简介
本项目为 COMM7340《AI for Digital Media》课程六人小组项目，主题是**基于深度学习的商品评论情绪演化分析与趋势预测**。

## 2. 课程要求对应说明
- Python 编程：全流程脚本位于 `src/`。
- 深度学习 + NLP：使用 Hugging Face `distilbert-base-uncased` 做三分类。
- 时序分析：周级情绪聚合 + PyTorch LSTM 预测。
- 双版本展示：Jupyter Notebook + Flask Web。

## 3. 数据集说明
- 数据集：Kaggle `snap/amazon-fine-food-reviews`
- 字段：`Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text`
- 自动下载：`src/01_download_data.py` 通过 `kagglehub` 下载。
- 回退机制：若 `data/raw/Reviews.csv` 已存在则直接加载本地文件。

## 4. 技术路线
A. 数据下载清洗与时间切分 → B. DistilBERT 情绪分类 → C. 生成评论情绪分数 → D. 构建周级时序 → E. LSTM 预测未来趋势 → F. 基线对比与可视化。

## 5. 文件结构
见仓库目录；核心模块在 `src/`，前端展示在 `app/`，输出在 `outputs/`。

## 6. 六人代码分工
1. 成员1（数据工程）：`src/01_download_data.py`, `src/02_preprocess_data.py`
2. 成员2（NLP建模）：`src/03_train_sentiment_model.py`
3. 成员3（评估与推理）：`src/04_evaluate_sentiment_model.py`, `src/05_generate_sentiment_scores.py`
4. 成员4（时序与LSTM）：`src/06_build_timeseries.py`, `src/07_train_lstm_forecaster.py`
5. 成员5（基线与可视化）：`src/08_baseline_models.py`, `src/09_visualize_results.py`
6. 成员6（Flask与集成）：`app/*`, `src/10_run_full_pipeline.py`

## 7. 环境安装方法
```bash
pip install -r requirements.txt
```

## 8. 数据自动下载说明
```bash
python src/01_download_data.py
```

## 9. Notebook 运行方法
打开并顺序执行：`notebooks/01_full_pipeline_sentiment_trend_forecasting.ipynb`。

## 10. 命令行脚本运行方法
```bash
python src/01_download_data.py
python src/02_preprocess_data.py
python src/03_train_sentiment_model.py --sample_size 100000 --epochs 3 --batch_size 16
python src/04_evaluate_sentiment_model.py
python src/05_generate_sentiment_scores.py --batch_size 64
python src/06_build_timeseries.py
python src/07_train_lstm_forecaster.py --epochs 100 --look_back 12
python src/08_baseline_models.py
python src/09_visualize_results.py
```
或一键运行：
```bash
python src/10_run_full_pipeline.py
```

## 11. Flask 网页版运行方法
```bash
cd app
python app.py
```

## 12. 主要结果输出位置
- `data/processed/cleaned_reviews.csv`
- `outputs/predictions/review_sentiment_scores.csv`
- `outputs/timeseries/weekly_sentiment.csv`
- `outputs/timeseries/future_forecast.csv`
- `outputs/metrics/sentiment_metrics.json`
- `outputs/metrics/lstm_metrics.json`
- `outputs/metrics/model_comparison.csv`
- `outputs/figures/*.png`

## 13. 报告和 PPT 可用内容
- 报告提纲：`reports/report_outline.md`
- 演示提纲：`reports/presentation_outline.md`
- 方法说明：`docs/project_methodology.md`
- 成员贡献：`docs/member_contributions.md`

## 14. 常见问题（FAQ）
- **没有 GPU**：支持 CPU，但 DistilBERT 训练会慢，建议 Colab T4。
- **没有 Kaggle 权限**：先配置 Kaggle 认证；或手动放置 `data/raw/Reviews.csv`。
- **模型文件不存在**：先运行训练脚本 `src/03_train_sentiment_model.py`。
- **图表空白/找不到**：先完成 `src/05~09` 生成输出文件。
