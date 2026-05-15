import pandas as pd
from utils import ensure_dirs

def add_sentiment(score):
    if score<=2:return 0
    if score==3:return 1
    return 2

def main():
    print('[INFO] Loading raw data...')
    df=pd.read_csv('data/raw/Reviews.csv')
    df=df.dropna(subset=['Text','Summary','Score','Time']).copy()
    df['Time']=pd.to_datetime(df['Time'],unit='s',errors='coerce')
    df=df.dropna(subset=['Time']).sort_values('Time')
    df['Year']=df['Time'].dt.year; df['Month']=df['Time'].dt.month
    df['Date']=df['Time'].dt.date.astype(str)
    df['Week']=df['Time'].dt.to_period('W').astype(str)
    df['sentiment_label']=df['Score'].apply(add_sentiment)
    n=len(df);a=int(n*0.7);b=int(n*0.85)
    df['split']='test'; df.iloc[:a,df.columns.get_loc('split')]='train'; df.iloc[a:b,df.columns.get_loc('split')]='val'
    ensure_dirs('data/processed')
    df.to_csv('data/processed/cleaned_reviews.csv',index=False)
    print('[INFO] Saved cleaned data to data/processed/cleaned_reviews.csv')

if __name__=='__main__':main()
