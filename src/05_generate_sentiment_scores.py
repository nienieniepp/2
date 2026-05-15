import argparse,pandas as pd, torch, numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from utils import ensure_dirs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--batch_size',type=int,default=64); args=ap.parse_args()
    df=pd.read_csv('data/processed/cleaned_reviews.csv')
    tok=DistilBertTokenizerFast.from_pretrained('models/sentiment_model'); m=DistilBertForSequenceClassification.from_pretrained('models/sentiment_model')
    m.eval(); device='cuda' if torch.cuda.is_available() else 'cpu'; m.to(device)
    probs=[]
    texts=df.Text.astype(str).tolist()
    for i in range(0,len(texts),args.batch_size):
        e=tok(texts[i:i+args.batch_size],truncation=True,padding=True,max_length=128,return_tensors='pt')
        with torch.no_grad(): l=m(input_ids=e['input_ids'].to(device),attention_mask=e['attention_mask'].to(device)).logits
        probs.append(torch.softmax(l,dim=-1).cpu().numpy())
    p=np.vstack(probs)
    out=df[['Time','Date','Week','ProductId','Score','Text','sentiment_label']].copy()
    out['predicted_label']=p.argmax(1); out['prob_negative']=p[:,0]; out['prob_neutral']=p[:,1]; out['prob_positive']=p[:,2]; out['sentiment_score']=out['prob_positive']
    ensure_dirs('outputs/predictions'); out.to_csv('outputs/predictions/review_sentiment_scores.csv',index=False)

if __name__=='__main__':main()
