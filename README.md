# Reinforcement Learning Stock Trading Bot

## Overview
This repository contains the implementation of Sai Vaka and Evan Zimmerman's CSE 592 final project. The project focuses on developing deep reinforcement learning (DRL) agents and ensemble strategies to optimize stock trading in dynamic and stochastic financial markets. The goal is to maximize portfolio returns while mitigating risks associated with market volatility and transaction costs.

### Key Features
- Implementation of five actor-critic DRL algorithms: PPO, A2C, DDPG, SAC, and TD3.
- Ensemble strategies for robust decision-making: simple averaging, weighted averaging, and a meta-agent.
- A custom stock trading environment built on OpenAI Gym, simulating realistic trading scenarios.
- Data preprocessing, including calculation of technical indicators (MACD, RSI, CCI, ADX) and normalization for model training.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Install dependencies using:
```bash
pip install -r requirements.txt
```

### Dataset

The dataset consists of daily stock data for the Dow Jones 30 stocks, spanning January 2009 to November 2024. It includes:

- **Features**: Open, High, Low, Close, Adjusted Close, Volume, and calculated technical indicators (MACD, RSI, CCI, ADX).
- The dataset is split into:
  - **Training Set**: 70% (2009–2016)
  - **Validation Set**: 15% (2017)
  - **Test Set**: 15% (2018–2022)

## Running the Project


### 1. Data Preprocessing (Only if you add more stock data)

The `data.py` script preprocesses the stock data by:

- Loading raw data from Yahoo Finance.
- Calculating technical indicators (e.g., MACD, RSI, CCI, ADX).
- Handling missing data and normalizing the dataset for stability.

Run the preprocessing script independently if needed:
```bash
python data.py
```

### 2. Train the Agents
Train the five reinforcement learning agents (PPO, A2C, DDPG, SAC, TD3) in the custom environment to maximize portfolio returns:
```bash
python train.py
```

### 3. Evaluate the Agents and Ensemble Strategies

Evaluate the performance of the trained agents and test ensemble strategies (simple averaging, weighted averaging, and meta-agent). The evaluation calculates metrics like cumulative returns and Sharpe ratio:

```bash
python eval.py
```

## Technical Details

### Actor-Critic Framework
The agents are based on the Actor-Critic framework:
- **Actor**: Outputs actions (e.g., buy/sell decisions) based on the state.
- **Critic**: Evaluates the actions and provides feedback using value estimates.

### Reinforcement Learning Agents
- **Proximal Policy Optimization (PPO)**: Limits policy updates for stability.
- **Advantage Actor-Critic (A2C)**: Synchronous updates using advantage estimates.
- **Deep Deterministic Policy Gradient (DDPG)**: Learns deterministic policies for continuous control.
- **Soft Actor-Critic (SAC)**: Balances exploration and exploitation via entropy regularization.
- **Twin Delayed DDPG (TD3)**: Improves DDPG by reducing overestimation bias.

### Ensemble Strategies
- **Simple Averaging**: Combines actions from all agents equally.
- **Weighted Averaging**: Adjusts weights dynamically based on agent performance.
- **Meta-Agent**: Uses a XGBoost model to predict optimal actions based on agent outputs.

---

## Results
- **Individual Agents**: Each DRL algorithm performed well in specific market conditions but had limitations.
- **Ensemble Strategies**: Demonstrated improved stability and adaptability, with better risk-adjusted returns compared to individual agents.
- **Metrics**: Evaluated using cumulative returns, volatility, and max drawdown 
---

## Future Work
- Integrate additional ensemble strategies (e.g., stacking).
- Expand the dataset to include global indexes and S&P 500 stocks.

