import os
import numpy as np
import pandas as pd
import time

import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError


def get_finance_data(ticker_symbol:str, source="yahoo",start="2021-01-01", end="2021-06-30", savedir="data") -> pd.DataFrame:
    """株価を記録したデータを取得します

    Args:
        ticker_symbol (str): Description of param1
        start (str): 期間はじめの日付, optional.
        end (str): 期間終わりの日付, optional.

    Returns:
        res: 株価データ

    """
    res = None
    filepath = os.path.join(savedir, f"{ticker_symbol}_{start}_{end}_historical.csv")
    os.makedirs(savedir, exist_ok=True)

    if not os.path.exists(filepath):
        try:
            time.sleep(5.0)  # MEMO: 連続アクセスを避ける
            res = web.DataReader(ticker_symbol, source, start=start, end=end)
            res.to_csv(filepath, encoding="utf-8-sig")
        except (RemoteDataError, KeyError):
            print(f"ticker_symbol ${ticker_symbol} が正しいか確認してください。")
    else:
        res = pd.read_csv(filepath, index_col="Date")
        res.index = pd.to_datetime(res.index)

    assert res is not None, "データ取得に失敗しました"
    return res


def get_rsi(close_prices: pd.Series, n=14):
    """RSI(相対力指数)を計算する
    RS＝（n日間の終値の上昇幅の平均）÷（n日間の終値の下落幅の平均）
    RSI= 100　-　（100　÷　（RS+1））

    参考文献：
      - https://info.monex.co.jp/technical-analysis/indicators/005.html
      - https://www.investopedia.com/terms/r/rsi.asp <-- 以下のコードの記号はこのページのものを使用

    Args:
        close_price (pd.Series): 終値の系列
        days (str): n日間, optional, default is 14.

    Returns:
        rsi(pd.Series): RSI
    """
    close_prices_diff = close_prices.diff(periods=1)[1:]
    fist_n_days_diff = close_prices_diff[: n + 1]
    previous_average_gain, previous_average_loss = 0, 0
    rsi = np.zeros_like(close_prices)

    for i in range(len(close_prices)):
        if i < n:

            previous_average_gain = fist_n_days_diff[fist_n_days_diff >= 0].sum() / n
            previous_average_loss = -fist_n_days_diff[fist_n_days_diff < 0].sum() / n
            rsi[i] = 100.0 - 100.0 / (1 + previous_average_gain / previous_average_loss)
        else:
            if (cpd_i := close_prices_diff[i - 1]) > 0:
                current_gain = cpd_i
                current_loss = 0.0
            else:
                current_gain = 0.0
                current_loss = -cpd_i

            previous_average_gain = (previous_average_gain * (n - 1) + current_gain) / n
            previous_average_loss = (previous_average_loss * (n - 1) + current_loss) / n
            rsi[i] = 100.0 - 100.0 / (1 + previous_average_gain / previous_average_loss)
    return rsi