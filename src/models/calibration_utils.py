"""
Calibration Analysis Module for classification models.

Provides tools to evaluate the reliability of predicted probabilities:
- Calibration curve (Reliability diagram)
- Brier score
- Expected Calibration Error (ECE)
- Comparison of calibration between models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


class CalibrationAnalyzer:
    """
    Calibration analysis class for classification models.
    
    Good calibration means that when the model predicts probability p,
    the actual rate of the event occurring is close to p.
    """
    
    def __init__(self, y_true: np.ndarray, y_proba: np.ndarray, model_name: str = 'Model'):
        """
        Initialize CalibrationAnalyzer.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            model_name: Model name.
        """
        self.y_true = np.array(y_true)
        self.y_proba = np.array(y_proba)
        self.model_name = model_name
        
    def compute_brier_score(self) -> float:
        """
        Compute Brier score.
        
        Brier score = mean((y_proba - y_true)^2)
        Lower values are better (0 = perfect, 1 = worst).
        
        Returns:
            Brier score.
        """
        return brier_score_loss(self.y_true, self.y_proba)
    
    def compute_calibration_curve(self, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve.
        
        Args:
            n_bins: Number of bins to divide probabilities.
            
        Returns:
            Tuple (fraction_of_positives, mean_predicted_value).
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_true, self.y_proba, n_bins=n_bins, strategy='uniform'
        )
        return fraction_of_positives, mean_predicted_value
    
    def compute_ece(self, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = sum(|bin_accuracy - bin_confidence| * bin_size) / total_samples
        
        Args:
            n_bins: Number of bins to divide probabilities.
            
        Returns:
            ECE score.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (self.y_proba >= bin_boundaries[i]) & (self.y_proba < bin_boundaries[i + 1])
            bin_size = bin_mask.sum()
            
            if bin_size > 0:
                bin_accuracy = self.y_true[bin_mask].mean()
                bin_confidence = self.y_proba[bin_mask].mean()
                ece += np.abs(bin_accuracy - bin_confidence) * bin_size
                
        ece /= len(self.y_true)
        return ece
    
    def compute_mce(self, n_bins: int = 10) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE = max(|bin_accuracy - bin_confidence|)
        
        Args:
            n_bins: Number of bins to divide probabilities.
            
        Returns:
            MCE score.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        max_error = 0.0
        
        for i in range(n_bins):
            bin_mask = (self.y_proba >= bin_boundaries[i]) & (self.y_proba < bin_boundaries[i + 1])
            bin_size = bin_mask.sum()
            
            if bin_size > 0:
                bin_accuracy = self.y_true[bin_mask].mean()
                bin_confidence = self.y_proba[bin_mask].mean()
                error = np.abs(bin_accuracy - bin_confidence)
                max_error = max(max_error, error)
                
        return max_error
    
    def get_calibration_summary(self, n_bins: int = 10) -> Dict:
        """
        Get summary of calibration metrics.
        
        Args:
            n_bins: Number of bins to divide probabilities.
            
        Returns:
            Dictionary containing metrics.
        """
        return {
            'model_name': self.model_name,
            'brier_score': self.compute_brier_score(),
            'ece': self.compute_ece(n_bins),
            'mce': self.compute_mce(n_bins),
            'mean_predicted': self.y_proba.mean(),
            'actual_positive_rate': self.y_true.mean()
        }
    
    def get_bin_analysis(self, n_bins: int = 10) -> pd.DataFrame:
        """
        Detailed analysis by each bin.
        
        Args:
            n_bins: Number of bins to divide probabilities.
            
        Returns:
            DataFrame containing bin analysis.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        results = []
        
        for i in range(n_bins):
            bin_mask = (self.y_proba >= bin_boundaries[i]) & (self.y_proba < bin_boundaries[i + 1])
            bin_size = bin_mask.sum()
            
            if bin_size > 0:
                bin_accuracy = self.y_true[bin_mask].mean()
                bin_confidence = self.y_proba[bin_mask].mean()
                calibration_error = bin_accuracy - bin_confidence
            else:
                bin_accuracy = np.nan
                bin_confidence = np.nan
                calibration_error = np.nan
                
            results.append({
                'Bin': f'{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}',
                'Count': bin_size,
                'Mean Predicted': bin_confidence if not np.isnan(bin_confidence) else 0,
                'Actual Positive Rate': bin_accuracy if not np.isnan(bin_accuracy) else 0,
                'Calibration Error': calibration_error if not np.isnan(calibration_error) else 0
            })
            
        return pd.DataFrame(results)
    
    def plot_calibration_curve(self, ax: plt.Axes = None, n_bins: int = 10) -> plt.Axes:
        """
        Plot calibration curve (Reliability diagram).
        
        Args:
            ax: Matplotlib axes.
            n_bins: Number of bins.
            
        Returns:
            Matplotlib axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        fraction_of_positives, mean_predicted_value = self.compute_calibration_curve(n_bins)
        brier = self.compute_brier_score()
        ece = self.compute_ece(n_bins)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label=f'{self.model_name} (Brier={brier:.4f}, ECE={ece:.4f})')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve (Reliability Diagram)')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_probability_histogram(self, ax: plt.Axes = None, n_bins: int = 20) -> plt.Axes:
        """
        Plot histogram of predicted probability distribution.
        
        Args:
            ax: Matplotlib axes.
            n_bins: Number of bins.
            
        Returns:
            Matplotlib axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.hist(self.y_proba[self.y_true == 0], bins=n_bins, alpha=0.7, 
                label='Negative (Actual)', color='red', density=True)
        ax.hist(self.y_proba[self.y_true == 1], bins=n_bins, alpha=0.7,
                label='Positive (Actual)', color='green', density=True)
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Probability Distribution - {self.model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax


def compare_calibration(
    y_true: np.ndarray,
    models_proba: Dict[str, np.ndarray],
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Compare calibration between multiple models.
    
    Args:
        y_true: True labels.
        models_proba: Dictionary {model_name: predicted_probabilities}.
        n_bins: Number of bins for analysis.
        
    Returns:
        DataFrame comparing models.
    """
    results = []
    
    for model_name, y_proba in models_proba.items():
        analyzer = CalibrationAnalyzer(y_true, y_proba, model_name)
        summary = analyzer.get_calibration_summary(n_bins)
        results.append(summary)
        
    df = pd.DataFrame(results)
    df = df.sort_values('brier_score')
    
    return df


def plot_multi_calibration_curves(
    y_true: np.ndarray,
    models_proba: Dict[str, np.ndarray],
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot calibration curves for multiple models on the same graph.
    
    Args:
        y_true: True labels.
        models_proba: Dictionary {model_name: predicted_probabilities}.
        n_bins: Number of bins.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=2)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_proba)))
    
    for (model_name, y_proba), color in zip(models_proba.items(), colors):
        analyzer = CalibrationAnalyzer(y_true, y_proba, model_name)
        fraction_of_positives, mean_predicted_value = analyzer.compute_calibration_curve(n_bins)
        brier = analyzer.compute_brier_score()
        
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                color=color, label=f'{model_name} (Brier={brier:.4f})')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Comparison')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    return fig
