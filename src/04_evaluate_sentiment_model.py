import json, pandas as pd, torch, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from utils import ensure_dirs

def main():
    df=pd.read_csv('data/processed/cleaned_reviews.csv'); te=df[df.split=='test'].head(5000)
    tok=DistilBertTokenizerFast.from_pretrained('models/sentiment_model'); m=DistilBertForSequenceClassification.from_pretrained('models/sentiment_model')
    m.eval(); device='cuda' if torch.cuda.is_available() else 'cpu'; m.to(device)
    enc=tok(te.Text.astype(str).tolist(),truncation=True,padding=True,max_length=128,return_tensors='pt')
    with torch.no_grad(): out=m(input_ids=enc['input_ids'].to(device),attention_mask=enc['attention_mask'].to(device)).logits
    pred=out.argmax(-1).cpu().numpy(); y=te.sentiment_label.values
    p,r,f1,_=precision_recall_fscore_support(y,pred,average='macro')
    metrics={'accuracy':float(accuracy_score(y,pred)),'precision':float(p),'recall':float(r),'macro_f1':float(f1),'classification_report':classification_report(y,pred,output_dict=True)}
    ensure_dirs('outputs/metrics','outputs/figures')
    json.dump(metrics,open('outputs/metrics/sentiment_metrics.json','w'),indent=2)
    cm=confusion_matrix(y,pred); plt.figure(figsize=(5,4)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues'); plt.savefig('outputs/figures/confusion_matrix.png',bbox_inches='tight')

if __name__=='__main__':main()
