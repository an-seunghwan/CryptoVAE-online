#%%
import pandas as pd
from datetime import datetime
import streamlit as st
#%%
def main():
    st.header("""
    How will the cryptocurrency price be tomorrow?
    """)
    
    data = pd.read_csv("./data/data.csv", index_col=0)
    
    st.write(f"""
    #### Settings
    - 7 cryptocurrencies: `BTC`, `ETH`, `XRP`, `ADA`, `ETC`, `BCH`, `XLM`
    - Train dataset window: {data.index[0]} ~ {data.index[-1]}
    - All cryptocurrency prices are scaled.
    - This result is based on the official implementation of 'Cryptocurrency Price Forecasting using Variational AutoEncoder with Versatile Quantile Modeling' (CIKM, 2024).
    """)
    
    today = datetime.strftime(
        pd.to_datetime(data.index[-1]), 
        '%Y-%m-%d %H:%M:%S')
    tomorrow = datetime.strftime(
        pd.to_datetime(data.index[-1]) + pd.Timedelta('24:00:00'), 
        '%Y-%m-%d %H:%M:%S')
    st.write("#### Cryptocurrency Price for Tomorrow")
    df = pd.read_csv("./assets/forecasting.csv", index_col=0)
    df["Today"] = data.iloc[-1].values
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- Today's price ({today})")
        st.dataframe(df[["names", "Today"]], height=300)
    with col2:
        st.write(f"- Tomorrow's price ({tomorrow})")
        st.dataframe(df[["names", "10%", "50%", "90%"]], height=300)
    
    st.write("#### Trends in 100-Day Forecasting Results")
    st.write("""
    - Tomorrow:
        - Orange downward triangle: 90% quantile
        - **x**: median
        - Orange upward triangle: 10% quantile
    """)
    st.image("assets/result.png")
    st.write("""
        - Dashed black line: **Actual** cryptocurrency prices
        - Green solid line: **Median** of predicted cryptocurrency prices
        - Blue area: **80% prediction interval** for cryptocurrency prices
    """)
    
    st.markdown("""---""")
    st.write("""
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
    
    st.markdown("""---""")
    st.write("""
    If you are interested in our results, please visit our GitHub!
    - [https://github.com/Optim-Lab/CryptoVAE](https://github.com/Optim-Lab/CryptoVAE)
    
    If you use this code or package, please cite our associated paper:
    ```
    
    ```
    
    Made by: 
    - Seunghwan An: <dkstmdghks79@uos.ac.kr>
    - Sungchul Hong: <shong@uos.ac.kr>
    """)

#%%
if __name__ == "__main__":
    main()