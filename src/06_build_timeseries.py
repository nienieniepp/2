import pandas as pd, matplotlib.pyplot as plt
from utils import ensure_dirs

def main():
    df=pd.read_csv('outputs/predictions/review_sentiment_scores.csv')
    wk=df.groupby('Week').agg(weekly_sentiment=('sentiment_score','mean'),review_count=('sentiment_score','size'),avg_score=('Score','mean')).reset_index()
    wk['ma12']=wk['weekly_sentiment'].rolling(12).mean()
    ensure_dirs('outputs/timeseries','outputs/figures'); wk.to_csv('outputs/timeseries/weekly_sentiment.csv',index=False)
    plt.figure(figsize=(10,4)); plt.plot(wk['weekly_sentiment'],label='weekly sentiment'); plt.plot(wk['ma12'],label='MA12'); plt.legend(); plt.savefig('outputs/figures/weekly_sentiment_trend.png',bbox_inches='tight')
    plt.figure(figsize=(10,3)); plt.bar(range(len(wk)),wk['review_count']); plt.savefig('outputs/figures/weekly_review_count.png',bbox_inches='tight')

if __name__=='__main__':main()
