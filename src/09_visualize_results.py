import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

def main():
    df=pd.read_csv('data/processed/cleaned_reviews.csv')
    plt.figure(); sns.countplot(x='sentiment_label',data=df); plt.savefig('outputs/figures/sentiment_distribution.png',bbox_inches='tight')
    wk=pd.read_csv('outputs/timeseries/weekly_sentiment.csv')
    plt.figure(figsize=(10,4)); plt.plot(wk['avg_score'],label='star'); plt.plot(wk['weekly_sentiment'],label='bert_sentiment'); plt.legend(); plt.savefig('outputs/figures/star_vs_sentiment.png',bbox_inches='tight')
    comp=pd.read_csv('outputs/metrics/model_comparison.csv')
    plt.figure(); sns.barplot(data=comp,x='model',y='rmse'); plt.savefig('outputs/figures/model_comparison.png',bbox_inches='tight')

if __name__=='__main__':main()
