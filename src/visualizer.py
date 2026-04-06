"""
Visualization Module for Bubble Detection

This module provides visualization tools for bubble signals and market data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime


class BubbleVisualizer:
    """
    Visualizes cryptocurrency market data and bubble detection signals.
    """
    
    def __init__(self, data: pd.DataFrame, signals: Optional[pd.DataFrame] = None):
        """
        Initialize the visualizer.
        
        Args:
            data (pd.DataFrame): Market data
            signals (pd.DataFrame, optional): Bubble signals data
        """
        self.data = data
        self.signals = signals
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
    
    def plot_price_with_signals(self, save_path: Optional[str] = None):
        """
        Plot price chart with bubble signals overlay.
        
        Args:
            save_path (str, optional): Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
        
        # Plot price
        ax1.plot(self.data.index, self.data['Close'], label='Price', color='blue', linewidth=2)
        
        if 'MA_30' in self.data.columns:
            ax1.plot(self.data.index, self.data['MA_30'], 
                    label='30-Day MA', color='orange', linestyle='--', alpha=0.7)
        
        if 'MA_90' in self.data.columns:
            ax1.plot(self.data.index, self.data['MA_90'], 
                    label='90-Day MA', color='green', linestyle='--', alpha=0.7)
        
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.set_title('Cryptocurrency Price with Bubble Signals', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot composite bubble signal
        if self.signals is not None and 'composite_score' in self.signals.columns:
            ax2.fill_between(self.signals.index, 0, self.signals['composite_score'], 
                            alpha=0.6, color='red', label='Bubble Signal')
            ax2.axhline(y=0.7, color='darkred', linestyle='--', 
                       label='High Risk Threshold', linewidth=2)
            ax2.axhline(y=0.5, color='orange', linestyle='--', 
                       label='Moderate Risk Threshold', linewidth=2)
            ax2.set_ylabel('Bubble Score', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylim(0, 1)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_individual_signals(self, save_path: Optional[str] = None):
        """
        Plot individual bubble detection signals.
        
        Args:
            save_path (str, optional): Path to save the figure
        """
        if self.signals is None:
            raise ValueError("No signals data available")
        
        signal_columns = [col for col in self.signals.columns if col != 'composite_score']
        n_signals = len(signal_columns)
        
        fig, axes = plt.subplots(n_signals, 1, figsize=(14, 3*n_signals), sharex=True)
        
        if n_signals == 1:
            axes = [axes]
        
        for i, signal in enumerate(signal_columns):
            axes[i].fill_between(self.signals.index, 0, self.signals[signal], 
                                alpha=0.6, label=signal.replace('_', ' ').title())
            axes[i].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            axes[i].set_ylabel('Signal Strength', fontsize=10)
            axes[i].set_ylim(0, 1)
            axes[i].legend(loc='upper left')
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date', fontsize=12)
        fig.suptitle('Individual Bubble Detection Signals', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_volatility_analysis(self, save_path: Optional[str] = None):
        """
        Plot volatility analysis.
        
        Args:
            save_path (str, optional): Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot returns
        ax1.plot(self.data.index, self.data['Returns'], 
                color='blue', alpha=0.6, linewidth=1)
        ax1.set_ylabel('Daily Returns', fontsize=12)
        ax1.set_title('Daily Returns Analysis', fontsize=12, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Plot volatility
        if 'Volatility_30' in self.data.columns:
            ax2.plot(self.data.index, self.data['Volatility_30'], 
                    color='red', linewidth=2, label='30-Day Volatility')
        if 'Volatility_7' in self.data.columns:
            ax2.plot(self.data.index, self.data['Volatility_7'], 
                    color='orange', linewidth=1, alpha=0.7, label='7-Day Volatility')
        
        ax2.set_ylabel('Volatility', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_title('Volatility Trends', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_bubble_periods(self, bubble_periods: List[Dict], save_path: Optional[str] = None):
        """
        Highlight detected bubble periods on price chart.
        
        Args:
            bubble_periods (list): List of bubble period dicts
            save_path (str, optional): Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot price
        ax.plot(self.data.index, self.data['Close'], 
               label='Price', color='blue', linewidth=2)
        
        # Highlight bubble periods
        for i, period in enumerate(bubble_periods):
            start = period['start_date']
            end = period['end_date']
            
            # Shade bubble period
            ax.axvspan(start, end, alpha=0.2, color='red', 
                      label='Bubble Period' if i == 0 else '')
            
            # Mark peak
            peak_date = period['peak_date']
            peak_price = period['peak_price']
            ax.scatter(peak_date, peak_price, color='red', s=100, 
                      zorder=5, marker='v')
            ax.text(peak_date, peak_price * 1.05, 
                   f"Peak\n{peak_price:.0f}", 
                   ha='center', fontsize=9)
        
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title('Detected Bubble Periods', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """
        Plot correlation heatmap of signals and features.
        
        Args:
            save_path (str, optional): Path to save the figure
        """
        # Select numeric columns for correlation
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col in [
            'Close', 'Volume', 'Returns', 'Volatility_30', 
            'RSI', 'Volume_Ratio', 'Price_MA30_Deviation'
        ]]
        
        if self.signals is not None:
            # Combine data and signals
            combined = pd.concat([
                self.data[feature_cols], 
                self.signals
            ], axis=1)
        else:
            combined = self.data[feature_cols]
        
        # Calculate correlation matrix
        corr = combined.corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Heatmap: Features and Signals', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_dashboard(self, bubble_periods: List[Dict] = None, 
                        save_path: Optional[str] = None):
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            bubble_periods (list, optional): List of detected bubble periods
            save_path (str, optional): Path to save the figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Price with signals
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.data.index, self.data['Close'], 
                label='Price', color='blue', linewidth=2)
        if 'MA_30' in self.data.columns:
            ax1.plot(self.data.index, self.data['MA_30'], 
                    label='30-Day MA', color='orange', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Price (USD)', fontsize=10)
        ax1.set_title('Price Chart', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Composite signal
        ax2 = fig.add_subplot(gs[1, :])
        if self.signals is not None and 'composite_score' in self.signals.columns:
            ax2.fill_between(self.signals.index, 0, self.signals['composite_score'], 
                            alpha=0.6, color='red')
            ax2.axhline(y=0.7, color='darkred', linestyle='--', linewidth=1)
            ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1)
        ax2.set_ylabel('Bubble Score', fontsize=10)
        ax2.set_title('Composite Bubble Signal', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 3. Volatility
        ax3 = fig.add_subplot(gs[2, 0])
        if 'Volatility_30' in self.data.columns:
            ax3.plot(self.data.index, self.data['Volatility_30'], 
                    color='red', linewidth=1.5)
        ax3.set_ylabel('Volatility', fontsize=10)
        ax3.set_xlabel('Date', fontsize=10)
        ax3.set_title('30-Day Volatility', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Volume
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.bar(self.data.index, self.data['Volume'], 
               color='green', alpha=0.6, width=1)
        ax4.set_ylabel('Volume', fontsize=10)
        ax4.set_xlabel('Date', fontsize=10)
        ax4.set_title('Trading Volume', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle('Cryptocurrency Bubble Detection Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
