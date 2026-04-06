"""
Data Fetcher Module for Cryptocurrency Market Data

This module provides functionality to fetch and preprocess cryptocurrency data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


class CryptoDataFetcher:
    """
    Fetches cryptocurrency market data from Yahoo Finance.
    
    Attributes:
        symbol (str): Cryptocurrency ticker symbol (e.g., 'BTC-USD')
        data (pd.DataFrame): Fetched market data
    """
    
    def __init__(self, symbol: str = 'BTC-USD'):
        """
        Initialize the data fetcher.
        
        Args:
            symbol (str): Cryptocurrency ticker symbol
        """
        self.symbol = symbol
        self.data = None
    
    def fetch_data(self, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None,
                   period: str = '2y') -> pd.DataFrame:
        """
        Fetch cryptocurrency data from Yahoo Finance.
        
        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            period (str): Period to fetch (e.g., '1y', '2y', '5y')
        
        Returns:
            pd.DataFrame: Market data with OHLCV columns
        """
        print(f"Fetching data for {self.symbol}...")
        
        try:
            if start_date and end_date:
                ticker = yf.Ticker(self.symbol)
                self.data = ticker.history(start=start_date, end=end_date)
            else:
                ticker = yf.Ticker(self.symbol)
                self.data = ticker.history(period=period)
            
            if self.data.empty:
                raise ValueError(f"No data fetched for {self.symbol}")
            
            print(f"Successfully fetched {len(self.data)} data points")
            return self.data
        
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise
    
    def calculate_features(self) -> pd.DataFrame:
        """
        Calculate technical indicators and features for bubble detection.
        
        Returns:
            pd.DataFrame: Data with additional feature columns
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
        
        df = self.data.copy()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Calculate moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['MA_90'] = df['Close'].rolling(window=90).mean()
        
        # Calculate volatility (rolling standard deviation)
        df['Volatility_7'] = df['Returns'].rolling(window=7).std()
        df['Volatility_30'] = df['Returns'].rolling(window=30).std()
        
        # Calculate price momentum
        df['Momentum_7'] = df['Close'] - df['Close'].shift(7)
        df['Momentum_30'] = df['Close'] - df['Close'].shift(30)
        
        # Calculate volume features
        df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_7']
        
        # Calculate RSI (Relative Strength Index)
        df['RSI'] = self._calculate_rsi(df['Close'], period=14)
        
        # Price acceleration (second derivative)
        df['Price_Acceleration'] = df['Returns'].diff()
        
        # Deviation from moving average (overextension indicator)
        df['Price_MA30_Deviation'] = (df['Close'] - df['MA_30']) / df['MA_30']
        
        self.data = df
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices (pd.Series): Price series
            period (int): RSI period
        
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the current data.
        
        Returns:
            pd.DataFrame: Current market data
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
        return self.data
