import argparse, pandas as pd, torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from utils import set_seed, ensure_dirs

class ReviewDS(Dataset):
    def __init__(self,texts,labels,tok,max_length):
        self.enc=tok(texts,truncation=True,padding=True,max_length=max_length)
        self.labels=labels
    def __len__(self): return len(self.labels)
    def __getitem__(self,idx):
        item={k:torch.tensor(v[idx]) for k,v in self.enc.items()}
        item['labels']=torch.tensor(int(self.labels[idx]))
        return item

def metrics(p):
    preds=p.predictions.argmax(-1)
    y=p.label_ids
    pr,re,f1,_=precision_recall_fscore_support(y,preds,average='macro')
    return {'accuracy':accuracy_score(y,preds),'precision':pr,'recall':re,'f1':f1}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sample_size',type=int,default=100000); ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--batch_size',type=int,default=16); ap.add_argument('--max_length',type=int,default=128);args=ap.parse_args()
    set_seed(42)
    if not torch.cuda.is_available(): print('[WARN] CPU mode detected, GPU is recommended for DistilBERT training.')
    df=pd.read_csv('data/processed/cleaned_reviews.csv')
    train=df[df.split=='train'].head(args.sample_size); val=df[df.split=='val'].head(max(1000,args.sample_size//5))
    tok=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model=DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=3)
    tr=ReviewDS(train.Text.astype(str).tolist(),train.sentiment_label.tolist(),tok,args.max_length)
    va=ReviewDS(val.Text.astype(str).tolist(),val.sentiment_label.tolist(),tok,args.max_length)
    ensure_dirs('models/sentiment_model','outputs/metrics')
    targs=TrainingArguments(output_dir='models/sentiment_model/checkpoints',num_train_epochs=args.epochs,per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=args.batch_size,eval_strategy='epoch',save_strategy='epoch',logging_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='f1',report_to='none')
    trainer=Trainer(model=model,args=targs,train_dataset=tr,eval_dataset=va,compute_metrics=metrics)
    trainer.train(); ev=trainer.state.log_history
    pd.DataFrame(ev).to_csv('outputs/metrics/sentiment_training_log.csv',index=False)
    trainer.save_model('models/sentiment_model'); tok.save_pretrained('models/sentiment_model')
    print('[INFO] Saved fine-tuned model.')

if __name__=='__main__':main()
