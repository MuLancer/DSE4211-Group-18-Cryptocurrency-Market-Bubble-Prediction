# Cryptocurrency Market Bubble Detection

**DSE4211 Group 18 Project**

A comprehensive Python-based system for detecting early signals of cryptocurrency market bubbles using technical analysis, statistical methods, and machine learning indicators.

## 🎯 Project Overview

This project implements a multi-factor bubble detection system that analyzes cryptocurrency market data to identify potential bubble formations before they burst. The system uses various technical indicators including:

- **Volatility Analysis**: Detects abnormal market volatility patterns
- **Price Momentum**: Identifies rapid, unsustainable price increases
- **Market Overextension**: Measures deviation from moving averages
- **Volume Anomalies**: Detects unusual trading volume spikes
- **RSI (Relative Strength Index)**: Identifies overbought conditions

These signals are combined into a composite score that provides an overall bubble risk assessment.

## 🚀 Features

- **Real-time Data Fetching**: Automatically retrieves cryptocurrency data from Yahoo Finance
- **Multi-factor Analysis**: Combines multiple bubble indicators for robust detection
- **Historical Bubble Identification**: Identifies past bubble periods in historical data
- **Risk Assessment**: Provides current bubble risk levels with interpretations
- **Comprehensive Visualizations**: Creates detailed charts and dashboards
- **Multi-cryptocurrency Support**: Analyze multiple cryptocurrencies simultaneously
- **Extensible Architecture**: Easy to add new indicators and detection methods

## 📋 Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/MuLancer/DSE4211-Group-18-Cryptocurrency-Market-Bubble-Prediction.git
cd DSE4211-Group-18-Cryptocurrency-Market-Bubble-Prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Analysis

Run a basic bubble detection analysis on Bitcoin:

```bash
python examples/basic_analysis.py
```

This will:
1. Fetch 2 years of BTC-USD data
2. Calculate technical indicators
3. Run bubble detection algorithms
4. Identify historical bubble periods
5. Assess current bubble risk
6. Generate visualization plots in the `output/` directory

### Multi-Cryptocurrency Analysis

Analyze multiple cryptocurrencies simultaneously:

```bash
python examples/multi_crypto_analysis.py
```

### Custom Analysis

Use the library in your own scripts:

```python
from src.data_fetcher import CryptoDataFetcher
from src.bubble_detector import BubbleDetector
from src.visualizer import BubbleVisualizer

# Fetch data
fetcher = CryptoDataFetcher(symbol='BTC-USD')
data = fetcher.fetch_data(period='2y')
data = fetcher.calculate_features()

# Detect bubbles
detector = BubbleDetector(data)
composite_score = detector.calculate_composite_score()

# Get current risk
risk = detector.get_current_bubble_risk()
print(f"Risk Level: {risk['risk_level']}")
print(f"Composite Score: {risk['composite_score']:.3f}")

# Identify bubble periods
bubble_periods = detector.identify_bubble_periods(threshold=0.6)

# Create visualizations
visualizer = BubbleVisualizer(data, detector.get_signals())
visualizer.create_dashboard(bubble_periods, save_path='dashboard.png')
```

## 📊 Output

The system generates several types of visualizations:

1. **Price with Signals**: Price chart overlaid with bubble detection signals
2. **Individual Signals**: Separate plots for each bubble indicator
3. **Volatility Analysis**: Returns and volatility trends
4. **Bubble Periods**: Historical bubble periods highlighted on price chart
5. **Correlation Heatmap**: Relationships between indicators
6. **Dashboard**: Comprehensive multi-panel view of all metrics

## 🔍 Bubble Detection Methodology

### Individual Signals

1. **Volatility Signal**: Detects periods of abnormally high volatility using z-score analysis
2. **Momentum Signal**: Identifies rapid price acceleration relative to historical trends
3. **Overextension Signal**: Measures price deviation from 30-day moving average
4. **Volume Signal**: Detects unusual trading volume spikes
5. **RSI Signal**: Identifies overbought conditions (RSI > 70)

### Composite Score

The composite score combines all individual signals using weighted averaging:
- Volatility: 25%
- Momentum: 20%
- Overextension: 20%
- Volume: 20%
- RSI: 15%

### Risk Levels

- **LOW** (0.0 - 0.3): Minimal bubble signals, market appears healthy
- **LOW-MODERATE** (0.3 - 0.5): Some indicators present but not alarming
- **MODERATE** (0.5 - 0.7): Elevated bubble indicators, monitor closely
- **HIGH** (0.7 - 1.0): Strong bubble signals, market may be overheated

## 📁 Project Structure

```
DSE4211-Group-18-Cryptocurrency-Market-Bubble-Prediction/
│
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── data_fetcher.py          # Data fetching and feature engineering
│   ├── bubble_detector.py       # Bubble detection algorithms
│   └── visualizer.py            # Visualization tools
│
├── examples/                     # Example scripts
│   ├── basic_analysis.py        # Basic usage example
│   └── multi_crypto_analysis.py # Multi-cryptocurrency analysis
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🎓 Educational Context

This project was developed as part of DSE4211 (Data Science for Economics) by Group 18. The goal is to apply data science techniques to financial markets, specifically:

- Time series analysis
- Feature engineering
- Anomaly detection
- Statistical modeling
- Data visualization

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.** 

- Do NOT use this as the sole basis for investment decisions
- Cryptocurrency markets are highly volatile and unpredictable
- Past performance does not guarantee future results
- Always conduct thorough research and consult financial advisors
- The authors are not responsible for any financial losses

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is for educational purposes as part of DSE4211 coursework.

## 👥 Authors

DSE4211 Group 18

## 📚 References

- Technical Analysis of Financial Markets
- Log-Periodic Power Law (LPPL) models for bubble detection
- Statistical methods in finance
- Yahoo Finance API documentation

## 🔮 Future Enhancements

Potential improvements for future versions:

- [ ] Implement LPPL (Log-Periodic Power Law) model fitting
- [ ] Add sentiment analysis from social media
- [ ] Include on-chain metrics (transaction volume, active addresses)
- [ ] Real-time monitoring and alerts
- [ ] Web-based dashboard
- [ ] Machine learning classification models
- [ ] Backtesting framework for strategy evaluation
- [ ] API for programmatic access

---

**Contact**: For questions about this project, please open an issue on GitHub.