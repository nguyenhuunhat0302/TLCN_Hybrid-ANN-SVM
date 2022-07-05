import pandas as pd
import numpy  as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import  mean_squared_error, mean_absolute_error
#DATA_URL = ('C:/Users/nhat0/Documents/TLCN/MyProject/data/VN_Index_Historical_Data(2011-2022)_C.csv')
### Đảo ngược DF
#df = df.reindex(index=df.index[::-1])
#df=df[::-1]
### xóa index
#df = df[::-1].reset_index(drop = True)
def dataF(DATA_URL):
    #df = pd.read_csv(DATA_URL)
    #df = df.reindex(index=df.index[::-1])
    #df['Date'] = pd.to_datetime(df['Date'])
    #df.set_index('Date', inplace=True)

    df = pd.read_csv(DATA_URL, index_col=0, parse_dates=True)
    #df['x_DATE'] = df['DATE'] + pd.DateOffset(days=180)
    #df = pd.read_csv('data/data_3.csv', parse_dates=['date'])
    #parse_dates=[['year', 'month', 'day']]
    #parse_dates={ 'date': ['year', 'month', 'day'] }
    #df = pd.read_csv(DATA_URL, index_col=0, parse_dates=['date'])
    #df['Date'] = pd.to_datetime(df['Date'], format='%Y-%M-%d')
    
    df=df[::-1]
    return df
def train_test_set(df,optionColums,split):
    df = df[[optionColums]]
    #df['Predict'] = df[[optionColums]].shift(-15)
    df['Predict'] = df.loc[:, (optionColums)].shift(-15)

    #df['Predict'] = df.loc[:, (optionColums)].shift(-15)
    X = np.array(df.drop(['Predict'],1))
    
    X = X[:-15]
    y = np.array(df['Predict'])
    y = y[:-15]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y= y.reshape(-1,1)
    y_scaled = scaler.fit_transform(y)
    #split = int(ratio_train/100 *len(X))
    # Train data set
    X_train = X_scaled[:split]
    y_train = y_scaled[:split]
    # Test data set
    X_test = X_scaled[split:]
    y_test = y_scaled[split:]
    return df,X_train,y_train,X_test,y_test,scaler

def calculate(x,y):
    def MAPE(Y_actual,Y_Predicted):
        mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
        return mape
    
    mape_ann = MAPE(x,y)
    RegScoreFun = r2_score(x,y)
    meanAbsoluteError_ann = mean_absolute_error(x,y)
    RMSE_ann = mean_squared_error(x, y)
    return mape_ann,RegScoreFun,meanAbsoluteError_ann,RMSE_ann