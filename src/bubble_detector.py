"""
Bubble Detection Module

This module implements various algorithms for detecting cryptocurrency market bubbles.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler


class BubbleDetector:
    """
    Detects potential bubble signals in cryptocurrency markets using multiple methods.
    
    Methods:
        - Volatility-based detection
        - Price momentum detection
        - LPPL (Log-Periodic Power Law) model fitting
        - Multi-factor composite score
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the bubble detector.
        
        Args:
            data (pd.DataFrame): Market data with calculated features
        """
        self.data = data.copy()
        self.bubble_signals = pd.DataFrame(index=data.index)
    
    def detect_volatility_bubble(self, threshold: float = 2.0) -> pd.Series:
        """
        Detect bubble signals based on abnormal volatility.
        
        High volatility combined with rapid price increases often signals bubbles.
        
        Args:
            threshold (float): Number of standard deviations for anomaly detection
        
        Returns:
            pd.Series: Bubble signal scores (0-1)
        """
        if 'Volatility_30' not in self.data.columns:
            raise ValueError("Volatility features not calculated. Run calculate_features first.")
        
        # Normalize volatility
        vol = self.data['Volatility_30'].fillna(0)
        vol_mean = vol.mean()
        vol_std = vol.std()
        
        # Calculate z-score
        vol_zscore = (vol - vol_mean) / vol_std if vol_std > 0 else vol * 0
        
        # Create signal (0-1 scale)
        signal = np.clip(vol_zscore / threshold, 0, 1)
        
        self.bubble_signals['volatility_signal'] = signal
        return signal
    
    def detect_momentum_bubble(self, short_window: int = 7, long_window: int = 30) -> pd.Series:
        """
        Detect bubble signals based on price momentum.
        
        Rapid price increases relative to historical averages signal potential bubbles.
        
        Args:
            short_window (int): Short-term momentum window
            long_window (int): Long-term momentum window
        
        Returns:
            pd.Series: Bubble signal scores (0-1)
        """
        # Calculate momentum ratio
        momentum_ratio = (
            self.data['Close'].pct_change(short_window) / 
            self.data['Close'].pct_change(long_window).abs()
        ).fillna(0)
        
        # Normalize to 0-1 scale using sigmoid
        signal = 1 / (1 + np.exp(-momentum_ratio * 2))
        
        self.bubble_signals['momentum_signal'] = signal
        return signal
    
    def detect_overextension_bubble(self, threshold: float = 0.3) -> pd.Series:
        """
        Detect bubble signals based on price overextension from moving average.
        
        Significant deviation from MA suggests overvaluation.
        
        Args:
            threshold (float): Deviation threshold (e.g., 0.3 = 30% above MA)
        
        Returns:
            pd.Series: Bubble signal scores (0-1)
        """
        if 'Price_MA30_Deviation' not in self.data.columns:
            raise ValueError("Price deviation not calculated. Run calculate_features first.")
        
        deviation = self.data['Price_MA30_Deviation'].fillna(0)
        
        # Positive deviations indicate overextension
        signal = np.clip(deviation / threshold, 0, 1)
        
        self.bubble_signals['overextension_signal'] = signal
        return signal
    
    def detect_volume_bubble(self, threshold: float = 2.0) -> pd.Series:
        """
        Detect bubble signals based on abnormal trading volume.
        
        Extreme volume spikes often accompany bubble peaks.
        
        Args:
            threshold (float): Volume ratio threshold
        
        Returns:
            pd.Series: Bubble signal scores (0-1)
        """
        if 'Volume_Ratio' not in self.data.columns:
            raise ValueError("Volume features not calculated. Run calculate_features first.")
        
        volume_ratio = self.data['Volume_Ratio'].fillna(1)
        
        # Normalize to 0-1 scale
        signal = np.clip((volume_ratio - 1) / threshold, 0, 1)
        
        self.bubble_signals['volume_signal'] = signal
        return signal
    
    def detect_rsi_bubble(self, overbought_threshold: float = 70) -> pd.Series:
        """
        Detect bubble signals based on RSI (Relative Strength Index).
        
        RSI values above 70 typically indicate overbought conditions.
        
        Args:
            overbought_threshold (float): RSI threshold for overbought condition
        
        Returns:
            pd.Series: Bubble signal scores (0-1)
        """
        if 'RSI' not in self.data.columns:
            raise ValueError("RSI not calculated. Run calculate_features first.")
        
        rsi = self.data['RSI'].fillna(50)
        
        # Map RSI to 0-1 scale where values > threshold = high signal
        signal = np.clip((rsi - overbought_threshold) / (100 - overbought_threshold), 0, 1)
        
        self.bubble_signals['rsi_signal'] = signal
        return signal
    
    def calculate_composite_score(self, weights: Dict[str, float] = None) -> pd.Series:
        """
        Calculate a composite bubble score from multiple signals.
        
        Args:
            weights (dict, optional): Weights for each signal component
        
        Returns:
            pd.Series: Composite bubble scores (0-1)
        """
        # Default equal weights
        if weights is None:
            weights = {
                'volatility_signal': 0.25,
                'momentum_signal': 0.20,
                'overextension_signal': 0.20,
                'volume_signal': 0.20,
                'rsi_signal': 0.15
            }
        
        # Run all detection methods if not already run
        if 'volatility_signal' not in self.bubble_signals.columns:
            self.detect_volatility_bubble()
        if 'momentum_signal' not in self.bubble_signals.columns:
            self.detect_momentum_bubble()
        if 'overextension_signal' not in self.bubble_signals.columns:
            self.detect_overextension_bubble()
        if 'volume_signal' not in self.bubble_signals.columns:
            self.detect_volume_bubble()
        if 'rsi_signal' not in self.bubble_signals.columns:
            self.detect_rsi_bubble()
        
        # Calculate weighted composite score
        composite = pd.Series(0, index=self.data.index)
        
        for signal_name, weight in weights.items():
            if signal_name in self.bubble_signals.columns:
                composite += self.bubble_signals[signal_name] * weight
        
        self.bubble_signals['composite_score'] = composite
        return composite
    
    def identify_bubble_periods(self, threshold: float = 0.6, min_duration: int = 5) -> List[Dict]:
        """
        Identify time periods where bubble signals exceed threshold.
        
        Args:
            threshold (float): Composite score threshold for bubble identification
            min_duration (int): Minimum duration (days) for a valid bubble period
        
        Returns:
            list: List of dicts with bubble period information
        """
        if 'composite_score' not in self.bubble_signals.columns:
            self.calculate_composite_score()
        
        scores = self.bubble_signals['composite_score']
        bubble_mask = scores > threshold
        
        bubble_periods = []
        in_bubble = False
        start_date = None
        
        for date, is_bubble in bubble_mask.items():
            if is_bubble and not in_bubble:
                # Start of bubble period
                in_bubble = True
                start_date = date
            elif not is_bubble and in_bubble:
                # End of bubble period
                duration = (date - start_date).days
                if duration >= min_duration:
                    max_score = scores.loc[start_date:date].max()
                    peak_date = scores.loc[start_date:date].idxmax()
                    
                    bubble_periods.append({
                        'start_date': start_date,
                        'end_date': date,
                        'duration_days': duration,
                        'max_score': max_score,
                        'peak_date': peak_date,
                        'peak_price': self.data.loc[peak_date, 'Close']
                    })
                
                in_bubble = False
                start_date = None
        
        # Handle case where bubble continues to end of data
        if in_bubble and start_date is not None:
            last_date = self.data.index[-1]
            duration = (last_date - start_date).days
            if duration >= min_duration:
                max_score = scores.loc[start_date:].max()
                peak_date = scores.loc[start_date:].idxmax()
                
                bubble_periods.append({
                    'start_date': start_date,
                    'end_date': last_date,
                    'duration_days': duration,
                    'max_score': max_score,
                    'peak_date': peak_date,
                    'peak_price': self.data.loc[peak_date, 'Close'],
                    'ongoing': True
                })
        
        return bubble_periods
    
    def get_current_bubble_risk(self) -> Dict:
        """
        Get the current bubble risk assessment based on latest data.
        
        Returns:
            dict: Current risk assessment with scores and interpretation
        """
        if 'composite_score' not in self.bubble_signals.columns:
            self.calculate_composite_score()
        
        latest_score = self.bubble_signals['composite_score'].iloc[-1]
        latest_signals = self.bubble_signals.iloc[-1].to_dict()
        
        # Determine risk level
        if latest_score >= 0.7:
            risk_level = "HIGH"
            interpretation = "Strong bubble signals detected. Market may be overheated."
        elif latest_score >= 0.5:
            risk_level = "MODERATE"
            interpretation = "Elevated bubble indicators. Monitor closely."
        elif latest_score >= 0.3:
            risk_level = "LOW-MODERATE"
            interpretation = "Some bubble indicators present but not alarming."
        else:
            risk_level = "LOW"
            interpretation = "Minimal bubble signals. Market appears healthy."
        
        return {
            'date': self.data.index[-1],
            'composite_score': latest_score,
            'risk_level': risk_level,
            'interpretation': interpretation,
            'individual_signals': latest_signals,
            'current_price': self.data['Close'].iloc[-1]
        }
    
    def get_signals(self) -> pd.DataFrame:
        """
        Get all calculated bubble signals.
        
        Returns:
            pd.DataFrame: All bubble signals
        """
        return self.bubble_signals
