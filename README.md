# SSM-Attention Candlestick Prediction System

A multi-source candlestick prediction system based on State Space Models (SSM) and Attention mechanisms, running entirely in the browser.

## 🚀 Features

- **Multi-source Data Fusion**: Combines VIX, TLT, GLD and other indicators to predict target candlesticks
- **State Space Expansion**: Uses SSM layers to project sequence features into high-dimensional state space
- **Attention Mechanism**: Learns correlations and importance between different data sources
- **Browser-based**: Built on TensorFlow.js, no backend server required
- **CSV Upload Support**: Upload your own yfinance-format CSV data files

## 📦 Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🏗️ Project Structure

```
├── index.html              # Main page
├── package.json            # Project configuration
├── vite.config.js          # Vite configuration
└── src/
    ├── main.js             # Entry file
    ├── app.js              # Main application logic
    ├── models/
    │   ├── model.js        # SSM-Attention model
    │   ├── ssmLayer.js     # State Space Model layer
    │   └── attentionLayer.js # Attention mechanism layer
    ├── utils/
    │   ├── dataProcessor.js # Data processing
    │   └── visualization.js # Visualization
    └── styles/
        └── main.css        # Stylesheet
```

## 🧠 Model Architecture

```
Input (Multi-source OHLCV) → Embedding → Positional Encoding → SSM Layer (State Expansion) → 
Multi-head Attention → Cross-source Attention → Prediction Head → OHLC Output
```

### Core Components

1. **SSM Layer**: Implements selective state space model similar to Mamba
   - Selective gating mechanism
   - State transition matrix
   - Sequence feature extraction

2. **Multi-head Attention**: Captures temporal dependencies
   - Self-attention mechanism
   - Causal masking (optional)

3. **Cross-source Attention**: Learns multi-source correlations
   - Source-level weight learning
   - Feature fusion

## 📊 Usage

1. Select data source mode (Mock data or CSV upload)
2. For CSV mode: Upload target file and optional related files
3. For Mock mode: Select prediction target (SPY/QQQ) and related sources (VIX, TLT, GLD, DXY)
4. Configure model parameters
5. Click "Load Data"
6. Click "Train Model"
7. Click "Predict"

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| Lookback Days | 30 | Historical window size |
| Forecast Days | 5 | Number of days to predict |
| SSM Hidden Dim | 32 | Model hidden layer dimension |
| State Dim | 16 | State space expansion dimension |
| Attention Heads | 2 | Number of attention heads |
| Training Epochs | 20 | Number of training epochs |
| Learning Rate | 0.001 | Adam optimizer learning rate |

## 📁 CSV Format

The system supports yfinance export format. Required columns:
```
Date, Open, High, Low, Close, Volume
```

You can obtain data using:
- Python `yfinance` library
- [Yahoo Finance](https://finance.yahoo.com) website

## 📝 License

MIT
