import argparse, json, numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LSTMModel(nn.Module):
    def __init__(self,h=50,layers=1,dropout=0.0):
        super().__init__()
        self.lstm=nn.LSTM(1,h,layers,batch_first=True,dropout=dropout if layers>1 else 0)
        self.fc=nn.Linear(h,1)
    def forward(self,x):
        o,_=self.lstm(x)
        return self.fc(o[:,-1,:])

def make_seq(data,lb):
    X,y=[],[]
    for i in range(len(data)-lb):
        X.append(data[i:i+lb]); y.append(data[i+lb])
    return np.array(X),np.array(y)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--look_back',type=int,default=12); args=ap.parse_args()
    s=pd.read_csv('outputs/timeseries/weekly_sentiment.csv')['weekly_sentiment'].values.reshape(-1,1)
    sc=MinMaxScaler(); d=sc.fit_transform(s); X,y=make_seq(d,args.look_back); cut=int(len(X)*0.8)
    Xtr,Xte,ytr,yte=X[:cut],X[cut:],y[:cut],y[cut:]
    Xtr=torch.tensor(Xtr,dtype=torch.float32); ytr=torch.tensor(ytr,dtype=torch.float32); Xte=torch.tensor(Xte,dtype=torch.float32)
    m=LSTMModel(); opt=torch.optim.Adam(m.parameters(),lr=0.001); lossf=nn.MSELoss()
    for _ in range(args.epochs):
        opt.zero_grad(); pred=m(Xtr); loss=lossf(pred,ytr); loss.backward(); opt.step()
    with torch.no_grad(): p=m(Xte).numpy()
    yt=sc.inverse_transform(yte); pt=sc.inverse_transform(p)
    rmse=float(np.sqrt(mean_squared_error(yt,pt))); mae=float(mean_absolute_error(yt,pt)); mape=float(np.mean(np.abs((yt-pt)/yt))*100) if np.all(yt!=0) else None
    json.dump({'rmse':rmse,'mae':mae,'mape':mape},open('outputs/metrics/lstm_metrics.json','w'),indent=2)
    torch.save({'model_state_dict':m.state_dict(),'look_back':args.look_back},'models/lstm_forecaster.pt')
    plt.figure(figsize=(8,4)); plt.plot(yt,label='actual'); plt.plot(pt,label='pred'); plt.legend(); plt.savefig('outputs/figures/lstm_forecast.png',bbox_inches='tight')
    seq_in=d[-args.look_back:].flatten().tolist(); fut=[]
    for _ in range(4):
        x=torch.tensor(np.array(seq_in[-args.look_back:]).reshape(1,args.look_back,1),dtype=torch.float32)
        n=float(m(x).item()); fut.append(n); seq_in.append(n)
    f=sc.inverse_transform(np.array(fut).reshape(-1,1)).flatten()
    pd.DataFrame({'week_ahead':[1,2,3,4],'forecast_sentiment':f}).to_csv('outputs/timeseries/future_forecast.csv',index=False)

if __name__=='__main__': main()
