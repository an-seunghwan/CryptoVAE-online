#%%
import pyupbit
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')
#%%
crypto_name_list = [
    "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-ETC", "KRW-BCH", "KRW-XLM"
]

top22_series_mktcap_list = []
for i in crypto_name_list:
    print("Downloading...:", i)
    top22_series_mktcap_list.append(
        pyupbit.get_ohlcv(i, interval="day", count=5000)['close']
    )

### pre-processing
df_top22_series = pd.concat(top22_series_mktcap_list, axis=1) 
df_top22_series.columns = crypto_name_list
df_crypto_indices = df_top22_series.ffill()
df_crypto_indices = df_top22_series.dropna(axis=0)

### scaling
df_crypto_indices["KRW-BTC"] = df_crypto_indices["KRW-BTC"]/1e7
df_crypto_indices["KRW-ETH"] = df_crypto_indices["KRW-ETH"]/1e6
df_crypto_indices["KRW-XRP"] = df_crypto_indices["KRW-XRP"]/1e2
df_crypto_indices["KRW-ADA"] = df_crypto_indices["KRW-ADA"]/1e2
df_crypto_indices["KRW-ETC"] = df_crypto_indices["KRW-ETC"]/1e4
df_crypto_indices["KRW-XLM"] = df_crypto_indices["KRW-XLM"]/1e2
df_crypto_indices["KRW-BCH"] = df_crypto_indices["KRW-BCH"]/1e5

### data save
df_crypto_indices.to_csv("./data/data.csv")
#%%