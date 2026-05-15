import pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evalm(y,p): return float(np.sqrt(mean_squared_error(y,p))), float(mean_absolute_error(y,p))

def main():
    y=pd.read_csv('outputs/timeseries/weekly_sentiment.csv')['weekly_sentiment'].values
    true=y[1:]; pers=y[:-1]; ma=np.array([y[max(0,i-4):i].mean() for i in range(1,len(y))])
    pr,pa=evalm(true,pers); mr,mae=evalm(true,ma)
    lstm=pd.read_json('outputs/metrics/lstm_metrics.json',typ='series')
    out=pd.DataFrame([['Persistence',pr,pa],['MovingAverage4',mr,mae],['LSTM',lstm.rmse,lstm.mae]],columns=['model','rmse','mae'])
    out.to_csv('outputs/metrics/model_comparison.csv',index=False)

if __name__=='__main__':main()
