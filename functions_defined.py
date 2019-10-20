

############################################## Functions ###############################################
from req_imports import *


#Moving Average  
def MA(df, n):  
    MA = pd.Series(pd.DataFrame.rolling(df['close'], n).mean(), name = 'MA_' + str(n))  
    df = df.join(MA)  
    return df

def VA(df, n):  
    VA = pd.Series(pd.DataFrame.rolling(df['volume'], n).mean(), name = 'VA_' + str(n))  
    df = df.join(VA)  
    return df
    
#Exponential Moving Average  
def EMA(df, n):  
    EMA = pd.Series(pd.DataFrame.ewm(df['close'], span = n, min_periods = n - 1).mean(), name = 'EMA_' + str(n))  
    df = df.join(EMA)  
    return df

#Average True Range  
def ATR(df, n):  
    i = 0  
    TR_l = [0]  
    while i < len(df)-1:  
        TR = max(df.loc[df.index[i + 1], 'high'], df.loc[df.index[i], 'close']) - min(df.loc[df.index[i + 1], 'low'], df.loc[df.index[i], 'close'])  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.DataFrame.ewm(TR_s, span = n, min_periods = n).mean(), name = 'ATR_' + str(n))  
    df['ATR_' + str(n)]=ATR.values 
    return df


#MACD, MACD Signal and MACD difference  
def MACD(df, n_fast, n_slow):  
    EMAfast = pd.Series(pd.DataFrame.ewm(df['close'], span = n_fast, min_periods = n_slow - 1).mean())  
    EMAslow = pd.Series(pd.DataFrame.ewm(df['close'], span = n_slow, min_periods = n_slow - 1).mean())  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(pd.DataFrame.ewm(MACD, span = 20, min_periods = 19).mean(), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df = df.join(MACD)  
    df = df.join(MACDsign)  
    df = df.join(MACDdiff)  
    return df


#Momentum  
def MOM(df, n):  
    M = pd.Series(df['close'].diff(n), name = 'Momentum_' + str(n))  
    df = df.join(M)  
    return df


#Stochastic oscillator %K  
def STOK(df):  
    SOk = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name = 'SO%k')  
    df = df.join(SOk)  
    return df

#Relative Strength Index  
def RSI(df, n):  
    i = 0  
    UpI = [0]  
    DoI = [0]  
    while i + 1 <= len(df)-1:  
        UpMove = df.loc[df.index[i + 1], 'high'] - df.loc[df.index[i], 'high']  
        DoMove = df.loc[df.index[i], 'low'] - df.loc[df.index[i + 1], 'low']  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.DataFrame.ewm(UpI, span = n, min_periods = n - 1).mean())  
    NegDI = pd.Series(pd.DataFrame.ewm(DoI, span = n, min_periods = n - 1).mean())  
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))  
    df['RSI_' + str(n)]=RSI.values
    return df

#Accumulation/Distribution  
def ACCDIST(df, n):  
    ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']  
    M = ad.diff(n - 1)  
    N = ad.shift(n - 1)  
    ROC = M / N  
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))  
    df = df.join(AD)  
    return df

#Force Index
def ForceIndex(df, n): 
    FI = pd.Series(df['close'].diff(n) * df['volume'], name = 'ForceIndex'+ str(n)) 
    df = df.join(FI) 
    return df

#drif independent volatility
def zy_vol(price_data, window, trading_periods=252, clean=True):

    log_ho = (price_data['high'] / price_data['open']).apply(np.log)
    log_lo = (price_data['low'] / price_data['open']).apply(np.log)
    log_co = (price_data['close'] / price_data['open']).apply(np.log)
    
    log_oc = (price_data['open'] / price_data['close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (price_data['close'] / price_data['close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1 + (window + 1) / (window - 1))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * m.sqrt(trading_periods)

    if clean:
        result.dropna()
        result=pd.Series(result, name = 'ZY_VOL_' + str(window)) 
        price_data = price_data.join(result)  
        return price_data
    else:
        result=pd.Series(result, name = 'ZY_VOL_' + str(window)) 
        price_data = price_data.join(result)  
        return price_data

#Normalize
def gaussian(x):
    x=x.replace([np.inf, -np.inf], np.nan)
    mean = x.mean()
    std = x.std()
    return (x - mean) / std, mean, std

#create lagged data and normalize
def normalize_and_lag(train,val,test,features,lags):
    global cols
    cols = []
    for lag in range(1, lags + 1):
        
        for  f in features:
            #print(lag)
            col = f+' _lag_%d' %lag
            if f in ['r', 'rs', 'd', 'u-d']:
                train[col] = train[f].shift(lag)
                val[col] = val[f].shift(lag)
                test[col] = test[f].shift(lag)
            else:
                train[col], mean, std = gaussian(train[f].shift(lag))
                val[col] = (val[f].shift(lag) - mean) / std
                test[col] = (test[f].shift(lag) - mean) / std
            cols.append(col)
            
    train.dropna(inplace=True)
    val.dropna(inplace=True)
    test.dropna(inplace=True)
    return train,val,test,cols



#Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = list(unique_labels(y_true, y_pred))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax