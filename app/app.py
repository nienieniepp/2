from __future__ import annotations
from pathlib import Path

import pandas as pd
import torch
from flask import Flask, render_template, request, send_file
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

app = Flask(__name__)
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "sentiment_model"
OUTPUTS = ROOT / "outputs"


def load_sentiment_model():
    if not (MODEL_DIR / "config.json").exists():
        return None, None
    tok = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tok, model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_text():
    result, msg = None, None
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        tok, model = load_sentiment_model()
        if model is None:
            msg = "未找到训练好的情绪模型。请先运行 python src/03_train_sentiment_model.py"
        elif not text:
            msg = "请输入评论文本。"
        else:
            enc = tok([text], truncation=True, padding=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
            labels = ["negative", "neutral", "positive"]
            result = {"label": labels[int(max(range(3), key=lambda i: probs[i]))], "probs": probs}
    return render_template("predict_text.html", result=result, msg=msg)


@app.route("/dashboard")
def dashboard():
    figures = [
        "sentiment_distribution.png",
        "weekly_sentiment_trend.png",
        "star_vs_sentiment.png",
        "confusion_matrix.png",
        "model_comparison.png",
    ]
    return render_template("dashboard.html", figures=figures)


@app.route("/forecast")
def forecast():
    forecast_path = OUTPUTS / "timeseries" / "future_forecast.csv"
    comparison_path = OUTPUTS / "metrics" / "model_comparison.csv"
    forecast_rows = pd.read_csv(forecast_path).to_dict("records") if forecast_path.exists() else []
    comparison_rows = pd.read_csv(comparison_path).to_dict("records") if comparison_path.exists() else []
    return render_template("forecast.html", forecast=forecast_rows, comparison=comparison_rows)


@app.route("/outputs/figures/<path:filename>")
def serve_output_figure(filename: str):
    path = OUTPUTS / "figures" / filename
    if not path.exists():
        return "Figure not found. Please run pipeline first.", 404
    return send_file(path)


if __name__ == "__main__":
    app.run(debug=True)
