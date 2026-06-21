# AI 股票預測系統（AI Stock Predictor）

結合 Machine Learning、金融資料分析與自動化驗證的台股預測系統。

使用 Python、XGBoost、FinMind、Yahoo Finance 與 FastAPI 建構，透過技術指標、法人籌碼、大盤資訊與美股領先指標，預測下一個交易日的股價方向，並提供預測驗證、歷史績效追蹤與 AI 解讀功能。

---

## 專案特色

### AI 漲跌預測

使用 XGBoost 建立預測模型，分析市場資料後預測：

* 隔日上漲或下跌方向
* 上漲機率與下跌機率
* 模型信心值
* 預測價格區間

---

### 多維度市場資訊

模型特徵包含：

#### 技術指標

* RSI
* KD 指標
* 移動平均線（MA5 / MA20 / MA60）
* MACD
* 報酬率（Return）
* 波動率（Volatility）

#### 法人籌碼

* 外資買賣超
* 投信買賣超
* 自營商買賣超

#### 大盤資訊

* 台股加權指數
* 大盤波動率
* 大盤報酬率

#### 美股領先指標

* NVIDIA（NVDA）
* 費城半導體指數（SOX）
* NASDAQ 100（QQQ）

---

### 預測驗證系統

每次預測結果皆會自動記錄：

* 預測日期
* 股票代號
* 預測方向
* 信心值
* 預測區間

系統於隔日收盤後自動驗證：

* 實際收盤價
* 實際漲跌方向
* 預測是否正確

建立完整預測歷史資料庫。

---

### AI 解讀功能

透過 Gemini API 將模型結果轉換為自然語言說明。

AI 僅根據模型輸出進行解讀，不進行新聞、法人動向或市場情緒推測，提升解讀一致性與可信度。

---

### 回測與績效分析

支援：

* Walk-Forward Backtesting
* Rolling Window Validation
* 多時間區間測試

測試期間：

* 30 天
* 90 天
* 180 天
* 365 天

並比較：

* AI 預測策略
* 買進持有策略

---

### 風險控制

支援：

* 停損機制
* 停利機制
* 交易成本模擬
* 最大回撤分析（Max Drawdown）

讓回測結果更貼近真實交易情境。

---

## 系統架構

輸入股票代號

↓

取得市場資料

↓

特徵工程

↓

XGBoost 預測

↓

AI 解讀

↓

預測結果儲存

↓

隔日自動驗證

↓

Dashboard 顯示績效

---

## 技術架構

### Backend

* Python
* FastAPI
* Jinja2

### Data Analysis

* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* XGBoost

### Financial Data

* FinMind
* Yahoo Finance

### Technical Analysis

* ta

### Frontend

* HTML
* CSS
* Chart.js

### AI

* Google Gemini API

---

## 未來規劃

### V5

* Explainable AI（Feature Importance）
* 模型解釋功能
* Dashboard 優化

### V6

* SHAP 單筆預測解釋
* Probability Calibration
* 多模型集成（Ensemble）

### V7

* LINE Bot 推播
* Make 自動化串接
* Email 預測通知
* 雲端部署

---

## 作者

林侑昕（Louis Lin）
