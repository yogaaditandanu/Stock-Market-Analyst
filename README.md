# 📈 IDX Pro Analyst System

A rule-based decision support system for Indonesian stocks, combining technical analysis, risk awareness, and lightweight machine learning forecasting.

This project is designed to **support decision quality**, not to predict markets or automate trading.

---

## 🚀 Overview

Many retail investors rely heavily on intuition, isolated indicators, or price predictions without sufficient structure or risk context.

**IDX Pro Analyst System** addresses this by providing:
- Transparent decision logic (BUY / HOLD / SELL)
- Clear separation between signal and execution
- Risk-aware insights before any action is considered

This is **not a trading bot** and **not financial advice**.

---

## 🧠 Core Features

### 1. Decision Engine (Rule-Based)
- Uses **RSI (14)** for momentum and overbought/oversold detection
- Uses **Moving Averages (MA50 & MA200)** for trend confirmation
- Two analysis modes:
  - **Decision Mode** → conservative, portfolio-safe
  - **Trading Mode** → execution-oriented with risk levels

Decision rules are **fully interpretable**, not a black box.

---

### 2. Risk Management First
- Automatic **Stop Loss** and **Take Profit** levels (Trading Mode)
- Clear distinction between:
  - **Signal** (analysis output)
  - **Execution** (user responsibility)
- Risk context is always shown before any forecast or action

> Risk is treated as a first-class component, not an afterthought.

---

### 3. ML Forecast (Supporting Insight)
- Uses **Linear Regression** on recent price trends
- Forecasts provide **directional context**, not certainty
- Predictions do **not** override decision logic

> Forecasts support decisions — they do not replace them.

---

### 4. Interactive Market Visualization
- Candlestick price charts
- Moving averages overlay
- RSI indicator panel
- Interactive exploration (zoom, hover, inspect)

Built to help users **understand market behavior**, not chase signals.

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** – web application framework
- **yFinance** – market data source
- **Pandas & NumPy** – data processing
- **Matplotlib / Plotly** – data visualization
- **Scikit-learn** – Linear Regression model

---

## 🌐 Deployment

The application is deployed as a **public Streamlit web app**.

- No login required
- Accessible for demonstration and portfolio review
- Designed for desktop and mobile viewing

🔗 Live Demo: *(add your Streamlit app link here)*

---

## 📂 Project Structure (Simplified)
├── app.py # Main Streamlit application
├── requirements.txt # Project dependencies
├── README.md # Project documentation


---

## ⚠️ Disclaimer

This project is created for **educational and portfolio purposes only**.

All investment decisions (buy/sell/hold) remain the sole responsibility of the user.  
Always conduct your own research (DYOR).

---

## 👤 Author

**Yoga Adi Tandanu**

- Background: Banking & Technology
- Focus: Data-driven decision making, risk-aware analytics, fintech systems

Feel free to explore, provide feedback, or connect.

