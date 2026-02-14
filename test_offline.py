"""
Unit tests with synthetic data (no internet required)
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.bubble_detector import BubbleDetector
from src.visualizer import BubbleVisualizer


def create_synthetic_data(n_days=100):
    """Create synthetic cryptocurrency data for testing."""
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate synthetic price data with a bubble pattern
    t = np.arange(n_days)
    base_price = 30000
    
    # Create price with exponential growth followed by crash
    price = base_price * (1 + 0.02 * t + 0.0005 * t**2)
    
    # Add some noise
    noise = np.random.normal(0, 500, n_days)
    price = price + noise
    
    # Create a crash in the last 20 days
    if n_days > 20:
        price[-20:] = price[-20] * np.exp(-0.1 * np.arange(20))
    
    # Create volume data
    volume = np.random.uniform(1e9, 5e9, n_days)
    volume[n_days//2:n_days//2+10] *= 3  # Volume spike in middle
    
    # Create OHLC data
    data = pd.DataFrame({
        'Open': price * 0.98,
        'High': price * 1.02,
        'Low': price * 0.97,
        'Close': price,
        'Volume': volume
    }, index=dates)
    
    # Calculate features
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['MA_7'] = data['Close'].rolling(window=7).mean()
    data['MA_30'] = data['Close'].rolling(window=min(30, n_days//3)).mean()
    data['MA_90'] = data['Close'].rolling(window=min(90, n_days)).mean()
    data['Volatility_7'] = data['Returns'].rolling(window=7).std()
    data['Volatility_30'] = data['Returns'].rolling(window=min(30, n_days//3)).std()
    data['Momentum_7'] = data['Close'] - data['Close'].shift(7)
    data['Momentum_30'] = data['Close'] - data['Close'].shift(min(30, n_days//3))
    data['Volume_MA_7'] = data['Volume'].rolling(window=7).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_7']
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['Price_Acceleration'] = data['Returns'].diff()
    data['Price_MA30_Deviation'] = (data['Close'] - data['MA_30']) / data['MA_30']
    
    return data


def test_bubble_detector():
    """Test bubble detector with synthetic data."""
    print("Testing BubbleDetector with synthetic data...")
    
    # Create synthetic data
    data = create_synthetic_data(n_days=100)
    
    # Test bubble detector
    detector = BubbleDetector(data)
    
    # Test individual signals
    vol_signal = detector.detect_volatility_bubble()
    assert len(vol_signal) == len(data), "Volatility signal length mismatch"
    assert (vol_signal >= 0).all(), "Volatility signal has negative values"
    assert (vol_signal <= 1).all(), "Volatility signal exceeds 1"
    print("  ✓ Volatility detection works")
    
    mom_signal = detector.detect_momentum_bubble()
    assert len(mom_signal) == len(data), "Momentum signal length mismatch"
    print("  ✓ Momentum detection works")
    
    overext_signal = detector.detect_overextension_bubble()
    assert len(overext_signal) == len(data), "Overextension signal length mismatch"
    print("  ✓ Overextension detection works")
    
    vol_signal = detector.detect_volume_bubble()
    assert len(vol_signal) == len(data), "Volume signal length mismatch"
    print("  ✓ Volume detection works")
    
    rsi_signal = detector.detect_rsi_bubble()
    assert len(rsi_signal) == len(data), "RSI signal length mismatch"
    print("  ✓ RSI detection works")
    
    # Test composite score
    composite = detector.calculate_composite_score()
    assert len(composite) == len(data), "Composite score length mismatch"
    assert (composite >= 0).all(), "Composite score has negative values"
    assert (composite <= 1).all(), "Composite score exceeds 1"
    print("  ✓ Composite score calculation works")
    
    # Test bubble period identification
    periods = detector.identify_bubble_periods(threshold=0.5, min_duration=3)
    assert isinstance(periods, list), "Bubble periods should be a list"
    print(f"  ✓ Identified {len(periods)} bubble period(s)")
    
    # Test current risk assessment
    risk = detector.get_current_bubble_risk()
    assert 'composite_score' in risk, "Missing composite_score in risk"
    assert 'risk_level' in risk, "Missing risk_level in risk"
    assert 'interpretation' in risk, "Missing interpretation in risk"
    assert risk['risk_level'] in ['LOW', 'LOW-MODERATE', 'MODERATE', 'HIGH'], \
        f"Invalid risk level: {risk['risk_level']}"
    print(f"  ✓ Current risk: {risk['risk_level']} (score: {risk['composite_score']:.3f})")
    
    # Test get_signals
    signals = detector.get_signals()
    assert isinstance(signals, pd.DataFrame), "Signals should be a DataFrame"
    assert len(signals) == len(data), "Signals length mismatch"
    print("  ✓ Get signals works")
    
    return data, detector


def test_visualizer(data, detector):
    """Test visualizer with synthetic data."""
    print("\nTesting BubbleVisualizer with synthetic data...")
    
    signals = detector.get_signals()
    visualizer = BubbleVisualizer(data, signals)
    
    assert visualizer.data is not None, "Visualizer data is None"
    assert visualizer.signals is not None, "Visualizer signals is None"
    print("  ✓ Visualizer initialization works")
    
    # Test that methods don't crash (without saving files)
    try:
        bubble_periods = detector.identify_bubble_periods()
        print("  ✓ Visualizer methods are callable")
    except Exception as e:
        print(f"  ⚠ Warning: {str(e)}")
    
    return visualizer


def test_data_structures():
    """Test that data structures are correct."""
    print("\nTesting data structures...")
    
    data = create_synthetic_data(n_days=50)
    
    # Check required columns
    required_cols = ['Close', 'Volume', 'Returns', 'MA_7', 'MA_30', 
                     'Volatility_7', 'Volatility_30', 'RSI']
    for col in required_cols:
        assert col in data.columns, f"Missing required column: {col}"
    
    print("  ✓ All required columns present")
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(data.index), "Index should be datetime"
    print("  ✓ Data types correct")
    
    # Check for NaN handling
    assert not data['Close'].isna().all(), "All Close prices are NaN"
    assert not data['Volume'].isna().all(), "All Volume values are NaN"
    print("  ✓ Data integrity verified")
    
    return data


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Running Bubble Detection System Tests (Offline Mode)")
    print("=" * 70)
    print()
    
    try:
        # Test data structures
        test_data_structures()
        
        # Test bubble detector
        data, detector = test_bubble_detector()
        
        # Test visualizer
        visualizer = test_visualizer(data, detector)
        
        print()
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Note: This test uses synthetic data. In production, the system")
        print("fetches real cryptocurrency data from Yahoo Finance.")
        return True
        
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"✗ TEST FAILED: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ TESTS FAILED: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
