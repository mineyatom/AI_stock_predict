# AI 股票預測系統

一個結合 Python、Machine Learning 與股市資料分析的股票預測系統，可輸入股票代號，自動抓取資料並預測下一交易日漲跌方向。



## 專案介紹

本專案主要透過：

* 技術指標
* 法人籌碼
* 大盤資訊
* 美股領先指標
* 新聞情緒（Experimental）

分析股票市場資訊，並使用 Machine Learning 模型預測：


明天可能上漲或下跌


除了預測外，也加入：

* 回測
* 風險控制
* 最大回撤分析

希望讓結果更接近真實交易情境。



## 專案功能

### 股票資料分析

自動抓取：

* 股票歷史價格
* 成交量
* 法人買賣超
* 大盤資訊
* 美股資料



### 技術指標分析

目前支援：

* RSI
* KD 指標
* 報酬率（Return）
* 波動率（Volatility）
* 成交量均線



### AI 漲跌預測

使用：
XGBoost

預測：

* 明日漲跌方向
* 上漲 / 下跌機率
* 信心值



### 多時間回測

支援：


30 / 90 / 180 / 365 天


並比較：

策略績效 vs 買進持有



### 風險控制

目前支援：

* 停損
* 停利
* 交易成本
* 最大回撤（Max Drawdown）



### 新聞情緒分析（Experimental）

使用：


Google News RSS


分析近期新聞內容。

目前透過：


關鍵字情緒判斷


分成：

`
利多 → +1
中性 → 0
利空 → -1




## 使用技術

### 後端開發

* Python
* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* XGBoost

### 股市資料

* Yahoo Finance
* FinMind

### 技術分析

* ta

### 視覺化

* Matplotlib



## 專案版本歷程

### V1 — 基礎模型

建立：

* 技術指標
* Logistic Regression
* Random Forest
* XGBoost

開始研究：


如何用股價資料建立 AI 模型



### V2 — 加入市場資訊

新增：

* 法人籌碼
* 大盤資訊
* 美股領先指標

開始發現：


不能只看個股本身
市場環境也會影響




### V3 — 回測與策略

加入：

* Walk-forward 回測
* 多時間回測
* 策略模式
* 最大回撤分析

開始研究：


準確率高 ≠ 一定賺錢



### V4 — 風險控制與新聞情緒

目前版本。

新增：

* Rolling Window
* 停損 / 停利
* 新聞情緒（Experimental）
* 信心值優化

也開始研究：


回測與風控的重要性




## 系統流程


輸入股票代號
        ↓
自動抓取股票資料
        ↓
技術指標與特徵工程
        ↓
XGBoost 預測漲跌
        ↓
回測策略績效
        ↓
顯示預測結果




## 未來規劃

* 多股票泛化模型（V5）
* FinBERT 新聞情緒分析
* 機率校正（Probability Calibration）
* 更完整風險控制



## 作者

林侑昕（Louis Lin）
