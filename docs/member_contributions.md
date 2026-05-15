# 六人代码分工（COMM7340）

## Member 1: Data Engineering
- 负责文件：`src/01_download_data.py`, `src/02_preprocess_data.py`
- 任务：自动下载、清洗、时间转换、情绪标签构造、时间切分。

## Member 2: NLP Deep Learning Model
- 负责文件：`src/03_train_sentiment_model.py`
- 任务：DistilBERT tokenizer/dataset/dataloader、微调、模型保存。

## Member 3: NLP Evaluation and Batch Inference
- 负责文件：`src/04_evaluate_sentiment_model.py`, `src/05_generate_sentiment_scores.py`
- 任务：accuracy/F1/precision/recall、混淆矩阵、批量概率推理。

## Member 4: Time Series and LSTM Forecasting
- 负责文件：`src/06_build_timeseries.py`, `src/07_train_lstm_forecaster.py`
- 任务：周级聚合、滑窗构建、LSTM、未来4周预测。

## Member 5: Baseline Models and Visualization
- 负责文件：`src/08_baseline_models.py`, `src/09_visualize_results.py`
- 任务：Persistence + MA4 基线、对比图与演示图。

## Member 6: Flask Web App and Project Integration
- 负责文件：`app/app.py`, `app/templates/*`, `app/static/css/style.css`, `src/10_run_full_pipeline.py`
- 任务：Web交互展示、一键运行与集成说明。
