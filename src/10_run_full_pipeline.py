import subprocess, sys
cmds=[[sys.executable,'src/01_download_data.py'],[sys.executable,'src/02_preprocess_data.py'],[sys.executable,'src/03_train_sentiment_model.py'],[sys.executable,'src/04_evaluate_sentiment_model.py'],[sys.executable,'src/05_generate_sentiment_scores.py'],[sys.executable,'src/06_build_timeseries.py'],[sys.executable,'src/07_train_lstm_forecaster.py'],[sys.executable,'src/08_baseline_models.py'],[sys.executable,'src/09_visualize_results.py']]
for c in cmds:
    print('[PIPELINE]',' '.join(c)); subprocess.run(c,check=True)
