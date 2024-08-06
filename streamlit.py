#%%
import pandas as pd
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
from ing_theme_matplotlib import mpl_style # pip install ing_theme_matplotlib 
import matplotlib as mpl
#%%
def main():
    st.header("""
    How will the cryptocurrency price be tomorrow?
    """)
    
    tab1, tab2 = st.tabs(["Tomorrow", "History"])
    
    data = pd.read_csv("./data/data.csv", index_col=0)
    
    tab1.write(f"""
    #### Settings
    - 7 cryptocurrencies: `BTC`, `ETH`, `XRP`, `ADA`, `ETC`, `BCH`, `XLM`
    - Train dataset window: {data.index[0]} ~ {data.index[-1]}
    - Our daily forecasting is based on the past 20 days.
    - This result is based on the official implementation of 'Cryptocurrency Price Forecasting using Variational AutoEncoder with Versatile Quantile Modeling' (CIKM, 2024).
        - We forecast 10%, 50%, 90% quantiles (interval estimation).
    """)
    
    scaling_dict = {
        "BTC": 1e7,
        "ETH": 1e6,
        "XRP": 1e2,
        "ADA": 1e2,
        "ETC": 1e4,
        "XLM": 1e2,
        "BCH": 1e5,
    }
    
    today = datetime.strftime(
        pd.to_datetime(data.index[-1]), 
        '%Y-%m-%d %H:%M:%S')
    tomorrow = datetime.strftime(
        pd.to_datetime(data.index[-1]) + pd.Timedelta('24:00:00'), 
        '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv("./assets/forecasting.csv", index_col=0)
    df["Today"] = data.iloc[-1].values
    df["Scale(KRW)"] = pd.DataFrame.from_dict(scaling_dict, orient="index").values
    df.index = df["names"]
    
    tab1.write("#### Cryptocurrency Price for Tomorrow")
    with tab1:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write(f"- Today's ({today})")
            st.dataframe(df[["Today", "Scale(KRW)"]], height=300)
        with col2:
            st.write(f"- Tomorrow's ({tomorrow})")
            st.dataframe(df[["10%", "50%", "90%", "Today", "Scale(KRW)"]], height=300)
    
    tab1.write("#### Trends in 100-Day Forecasting Results")
    tab1.write("""
    - Tomorrow:
        - Orange downward triangle: 90% Quantile
        - **x**: Median
        - Orange upward triangle: 10% Quantile
    """)
    tab1.image("assets/result.png")
    tab1.write("""
    - Train Dataset:
        - Dashed black line: **Actual** cryptocurrency prices
        - Green solid line: **Median** of predicted cryptocurrency prices
        - Blue area: **80% prediction interval** for cryptocurrency prices
    """)
    
    tab1.markdown("""---""")
    tab1.write("""
    ###### Disclaimer
        The cryptocurrency price prediction model provided on this website is for informational purposes only and does not constitute financial advice. 
        The predictions and analysis are based on historical data and do not guarantee future performance. 
        Investing in cryptocurrencies involves significant risk, and you should be aware that you could lose some or all of your investment.
        We are not responsible for any financial losses that may result from your investment decisions. 
        You are solely responsible for your own investment choices.         
    
    ###### 책임 고지
        이 웹사이트에서 제공하는 가상화폐 가격 예측 모형은 정보 제공 목적으로만 사용되며 재정적 조언을 구성하지 않습니다. 
        예측 및 분석은 과거 데이터를 기반으로 하며 미래 성과를 보장하지 않습니다. 
        가상화폐 투자에는 상당한 위험이 따르며, 투자한 금액의 일부 또는 전부를 잃을 수 있음을 인지해야 합니다. 
        본 웹사이트와 제작자는 귀하의 투자 결정으로 인해 발생할 수 있는 재정적 손실에 대해 책임지지 않습니다. 
        투자의 책임은 전적으로 귀하에게 있습니다.
    """)
    
    tab1.markdown("""---""")
    tab1.write("""
    If you are interested in our results, please visit our GitHub!
    - [https://github.com/Optim-Lab/CryptoVAE](https://github.com/Optim-Lab/CryptoVAE)
    
    If you use this code or package, please cite our associated paper:
    ```
    
    ```
    
    Made by: 
    - Seunghwan An: <dkstmdghks79@uos.ac.kr>
    - Sungchul Hong: <shong@uos.ac.kr>
    """)
    
    history = pd.read_csv("./assets/history.csv", index_col=0).iloc[-100:]
    times = history["time"].unique()
    cryptos = history["names"].unique()
    
    mpl.rcParams["figure.dpi"] = 200
    mpl_style(dark=False)
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    SMALL_SIZE = 28
    BIGGER_SIZE = 32
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    tab2.write("#### Trends in Real Forecasting History")
    fig, ax = plt.subplots(4, 2, figsize=(35, 40))
    for j in range(len(cryptos)):
        history_ = history.loc[history["names"] == cryptos[j]]
        history_['time'] = history_['time'].apply(lambda x: x.split()[0])
        data_ = data.iloc[-len(times)+1:][f"KRW-{cryptos[j]}"].reset_index()
    
        ax.flatten()[j].plot(
            history_['time'], history_['50%'] * scaling_dict.get(cryptos[j]),
            label="Median", color='green', linewidth=6)
        ax.flatten()[j].fill_between(
            history_['time'], 
            history_['10%'] * scaling_dict.get(cryptos[j]), 
            history_['90%'] * scaling_dict.get(cryptos[j]), 
            color=cols[3], alpha=0.5, label=r'80% interval')
        ax.flatten()[j].plot(
            data_.index, data_[f"KRW-{cryptos[j]}"] * scaling_dict.get(cryptos[j]), 
            label=f"KRW-{cryptos[j]}", color="black", linewidth=6, linestyle='--')
        ax.flatten()[j].set_ylabel(f"{cryptos[j]}")
        # ax.flatten()[j].set_xlabel("days")
        ax.flatten()[j].legend(loc="lower left")
        ax.flatten()[j].tick_params(axis='x', rotation=45)
    ax[-1, -1].axis('off')
    plt.tight_layout()
    tab2.pyplot(fig)
    tab2.write("""
    - Train Dataset:
        - Dashed black line: **Actual** cryptocurrency prices
        - Green solid line: **Median** of predicted cryptocurrency prices
        - Blue area: **80% prediction interval** for cryptocurrency prices
    """)
#%%
if __name__ == "__main__":
    main()