from pathlib import Path
import pandas as pd, torch
from flask import Flask, render_template, request
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app=Flask(__name__)
ROOT=Path(__file__).resolve().parents[1]
model_dir=ROOT/'models/sentiment_model'

def try_model():
    if not model_dir.exists() or not (model_dir/'config.json').exists(): return None,None
    tok=DistilBertTokenizerFast.from_pretrained(model_dir); m=DistilBertForSequenceClassification.from_pretrained(model_dir); m.eval(); return tok,m

@app.route('/')
def index(): return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    result=None; msg=None
    if request.method=='POST':
        text=request.form.get('text','')
        tok,m=try_model()
        if m is None: msg='未找到训练模型，请先运行训练脚本。'
        else:
            enc=tok([text],truncation=True,padding=True,max_length=128,return_tensors='pt')
            with torch.no_grad(): p=torch.softmax(m(**enc).logits,dim=-1)[0].tolist()
            labels=['negative','neutral','positive']; result={'label':labels[int(max(range(3),key=lambda i:p[i]))],'probs':p}
    return render_template('predict_text.html',result=result,msg=msg)

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/forecast')
def forecast():
    f=ROOT/'outputs/timeseries/future_forecast.csv'; c=ROOT/'outputs/metrics/model_comparison.csv'
    forecast=pd.read_csv(f).to_dict('records') if f.exists() else []
    comp=pd.read_csv(c).to_dict('records') if c.exists() else []
    return render_template('forecast.html',forecast=forecast,comparison=comp)

if __name__=='__main__': app.run(debug=True)
