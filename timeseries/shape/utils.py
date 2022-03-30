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


def DDTW(Q, C):
    """
    Args:
        Q (np.array or list): 一つ目の波形
        C (np.array or list): 二つ目の波形

    Returns:
        γ_mat (np.array): DDTWを計算するための行列
        arrows (np.array): 各時点で←・↙︎・↓のどのマスが最小だったかを示す記号を保存する行列
        ddtw (float): DDTW
    """
    Q, C = np.array(Q), np.array(C)
    assert Q.shape[0] > 3, "一つ目の波形のフォーマットがおかしいです。"
    assert C.shape[0] > 3, "二つ目の波形のフォーマットがおかしいです。"

    # 3.1 Algorithm details の式
    def _Dq(q):
        return ((q[1] - q[0]) + (q[2] - q[0]) / 2) / 2

    # 二つの時点間の距離
    def _γ(x, y):
        return abs(_Dq(x) - _Dq(y))

    # 各変数
    n, m = Q.shape[0] - 2, C.shape[0] - 2
    γ_mat = np.zeros((n, m))
    arrows = np.array(np.zeros((n, m)), dtype=str)  # 可視化用の行列でDDTWの値とは無関係

    # 一番左下のスタート地点
    γ_mat[0, 0] = _γ(Q[0:3], C[0:3])

    # 一列目を計算
    for i in range(1, n):
        γ_mat[i, 0] = γ_mat[i - 1, 0] + _γ(Q[i - 1 : i + 2], C[0:3])
        arrows[i, 0] = "↓"

    # 一行目を計算
    for j in range(1, m):
        γ_mat[0, j] = γ_mat[0, j - 1] + _γ(Q[0:3], C[j - 1 : j + 2])
        arrows[0, j] = "←"

    # 残りのマスを計算
    for i in range(1, n):
        for j in range(1, m):
            # DDTWを求めるためのマトリクスを埋める
            d_ij = _γ(Q[i - 1 : i + 2], C[j - 1 : j + 2])
            γ_mat[i, j] = d_ij + np.min(
                [γ_mat[i - 1, j - 1], γ_mat[i - 1, j], γ_mat[i, j - 1]]
            )

            # 矢印を書くための行列(DDTWの値とは関係無い処理)
            if (
                square_index := np.argmin(
                    [γ_mat[i - 1, j - 1], γ_mat[i - 1, j], γ_mat[i, j - 1]]
                )
            ) == 0:
                arrows[i, j] = "↙︎"
            elif square_index == 1:
                arrows[i, j] = "↓"
            elif square_index == 2:
                arrows[i, j] = "←"

    return γ_mat, arrows, γ_mat[n - 1, m - 1]