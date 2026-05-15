from __future__ import annotations
import subprocess
import sys


def main() -> None:
    commands = [
        [sys.executable, "src/01_download_data.py"],
        [sys.executable, "src/02_preprocess_data.py"],
        [sys.executable, "src/03_train_sentiment_model.py", "--sample_size", "100000", "--epochs", "3", "--batch_size", "16"],
        [sys.executable, "src/04_evaluate_sentiment_model.py"],
        [sys.executable, "src/05_generate_sentiment_scores.py", "--batch_size", "64"],
        [sys.executable, "src/06_build_timeseries.py"],
        [sys.executable, "src/07_train_lstm_forecaster.py", "--epochs", "100", "--look_back", "12"],
        [sys.executable, "src/08_baseline_models.py"],
        [sys.executable, "src/09_visualize_results.py"],
    ]
    for cmd in commands:
        print("[PIPELINE]", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
