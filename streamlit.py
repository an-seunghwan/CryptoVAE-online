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