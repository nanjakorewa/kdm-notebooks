### 投資信託で効率的フロンティアを計算してみる

- `2014091602`: AB･米国成長株投信Dコース(H無) 予想分配金
  - https://www.wealthadvisor.co.jp/FundData/SnapShot.do?fnc=2014091602
- `2018103105`:eMAXIS Slim全世界株式(オール･カントリー) 
  - https://www.wealthadvisor.co.jp/FundData/SnapShot.do?fnc=2018103105
- `2015042708`: ニッセイ TOPIXインデックスファンド
  - https://www.wealthadvisor.co.jp/FundData/SnapShot.do?fnc=2015042708
- `2019092601`: SBI・V・S&P500インデックス・ファンド
  - https://www.wealthadvisor.co.jp/FundData/SnapShot.do?fnc=2019092601
- `2018070301`: eMAXIS Slim米国株式(S&P500)
  - https://www.wealthadvisor.co.jp/FundData/SnapShot.do?fnc=2018070301
- `2011020701`: 三菱UFJ 純金ファンド（←保有していないですが金価格の参考として）
  - https://www.wealthadvisor.co.jp/FundData/SnapShot.do?fnc=2011020701


### データの読み込み
[WEALTH ADVISER](https://www.wealthadvisor.co.jp)から手動で取得した週次のcsvデータを使用して、効率的フロンティアを計算してみます。





```python
import os

import pandas as pd

データ保存先ディレクトリ = "C://Users//nanja-win-ms//Dropbox//PC//Downloads//"
投資信託一覧 = {
    "2018103105": "eMAXIS Slim全世界株式(オール･カントリー)",
    "2015042708": "ニッセイ TOPIXインデックスファンド",
    "2019092601": "SBI・V・S&P500インデックス・ファンド",
    "2018070301": "eMAXIS Slim米国株式(S&P500)",
    "2011020701": "三菱UFJ 純金ファンド",
}

投資信託リターン = {}

for 投資信託ID, 投資信託名 in 投資信託一覧.items():
    月次リターンファイル名 = [
        c
        for c in os.listdir(os.path.join(データ保存先ディレクトリ, 投資信託ID))
        if c.startswith("基準価額")
    ][0]
    投資信託リターン[投資信託ID] = pd.read_csv(
        os.path.join(データ保存先ディレクトリ, 投資信託ID, 月次リターンファイル名), encoding="cp932"
    )
    投資信託リターン[投資信託ID].columns = ["日付", 投資信託名]
```

### データの整形

ｘ＝日付、ｙ＝銘柄名のデータを作成します。投資信託でなく株式でも可能です。


```python
ポートフォリオ = None

for 投資信託ID in 投資信託一覧.keys():
    if ポートフォリオ is None:
        ポートフォリオ = 投資信託リターン[投資信託ID]
    else:
        ポートフォリオ = pd.merge(ポートフォリオ, 投資信託リターン[投資信託ID], on="日付")


ポートフォリオ.index = pd.to_datetime(ポートフォリオ["日付"], format="%Y%m%d")
ポートフォリオ.drop("日付", axis=1, inplace=True)

ポートフォリオ
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eMAXIS Slim全世界株式(オール･カントリー)</th>
      <th>ニッセイ TOPIXインデックスファンド</th>
      <th>SBI・V・S&amp;P500インデックス・ファンド</th>
      <th>eMAXIS Slim米国株式(S&amp;P500)</th>
      <th>三菱UFJ 純金ファンド</th>
    </tr>
    <tr>
      <th>日付</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-10</th>
      <td>11828</td>
      <td>11720</td>
      <td>11203</td>
      <td>12185</td>
      <td>13827</td>
    </tr>
    <tr>
      <th>2020-01-17</th>
      <td>12036</td>
      <td>11723</td>
      <td>11429</td>
      <td>12428</td>
      <td>14063</td>
    </tr>
    <tr>
      <th>2020-01-24</th>
      <td>11932</td>
      <td>11689</td>
      <td>11381</td>
      <td>12378</td>
      <td>14008</td>
    </tr>
    <tr>
      <th>2020-01-31</th>
      <td>11667</td>
      <td>11379</td>
      <td>11189</td>
      <td>12168</td>
      <td>14085</td>
    </tr>
    <tr>
      <th>2020-02-07</th>
      <td>11987</td>
      <td>11703</td>
      <td>11490</td>
      <td>12496</td>
      <td>14242</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>20899</td>
      <td>17484</td>
      <td>22229</td>
      <td>24281</td>
      <td>23054</td>
    </tr>
    <tr>
      <th>2024-01-05</th>
      <td>20972</td>
      <td>17684</td>
      <td>22285</td>
      <td>24342</td>
      <td>23270</td>
    </tr>
    <tr>
      <th>2024-01-12</th>
      <td>21283</td>
      <td>18428</td>
      <td>22763</td>
      <td>24871</td>
      <td>23226</td>
    </tr>
    <tr>
      <th>2024-01-19</th>
      <td>21536</td>
      <td>18543</td>
      <td>23219</td>
      <td>25369</td>
      <td>23562</td>
    </tr>
    <tr>
      <th>2024-01-26</th>
      <td>21916</td>
      <td>18450</td>
      <td>23699</td>
      <td>25888</td>
      <td>23437</td>
    </tr>
  </tbody>
</table>
<p>212 rows × 5 columns</p>
</div>



### 可視化


```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.lineplot(data=ポートフォリオ)
```




    <Axes: xlabel='日付'>




    
![png](pyPortfolioOpt_files/pyPortfolioOpt_6_1.png)
    



```python
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

mu = expected_returns.mean_historical_return(ポートフォリオ)
S = risk_models.sample_cov(ポートフォリオ)
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)
```

    Expected annual return: 115.5%
    Annual volatility: 34.2%
    Sharpe Ratio: 3.32
    




    (1.1546507705298557, 0.34188826020611157, 3.318776637272708)




```python
from pypfopt import CLA, plotting

cla = CLA(mu, S)
cla.max_sharpe()
cla.portfolio_performance(verbose=True)

plt.figure(figsize=(10, 5))
plotting.plot_efficient_frontier(cla, show_assets=True, points=50, show_tickers=True)
```

    Expected annual return: 115.1%
    Annual volatility: 34.1%
    Sharpe Ratio: 3.32
    




    <Axes: xlabel='Volatility', ylabel='Return'>




    
![png](pyPortfolioOpt_files/pyPortfolioOpt_8_2.png)
    


### 値動きの相関
値動きの変化で相関を計算してみます。


```python
sns.heatmap(ポートフォリオ.diff().dropna().corr(), annot=True, fmt="1.4f")
```




    <Axes: >




    
![png](pyPortfolioOpt_files/pyPortfolioOpt_10_1.png)
    

