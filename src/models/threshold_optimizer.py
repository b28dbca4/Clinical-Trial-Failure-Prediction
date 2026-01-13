"""
Threshold Optimization Module for binary classification problems.

Provides threshold optimization strategies including:
- Maximize F1-Score
- Ensure minimum Recall
- Youden's J statistic
- Cost-sensitive optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix
)


class ThresholdOptimizer:
    """
    Threshold optimization class for binary classification models.
    
    Attributes:
        y_true: True labels.
        y_proba: Predicted probabilities for the positive class.
    """
    
    def __init__(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Initialize ThresholdOptimizer.
        
        Args:
            y_true: True labels (0 or 1).
            y_proba: Predicted probabilities for the positive class.
        """
        self.y_true = np.array(y_true)
        self.y_proba = np.array(y_proba)
        self.thresholds = np.linspace(0.01, 0.99, 99)
        
    def _compute_metrics_at_threshold(self, threshold: float) -> Dict:
        """
        Compute metrics at a specific threshold.
        
        Args:
            threshold: Decision threshold.
            
        Returns:
            Dictionary containing metrics.
        """
        y_pred = (self.y_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred, labels=[0, 1]).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
    
    def find_optimal_f1(self) -> Dict:
        """
        Find threshold that optimizes F1-Score.
        
        Returns:
            Dictionary containing optimal threshold and metrics.
        """
        best_f1 = 0
        best_result = None
        
        for threshold in self.thresholds:
            metrics = self._compute_metrics_at_threshold(threshold)
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_result = metrics
                
        best_result['strategy'] = 'max_f1'
        return best_result
    
    def find_threshold_for_recall(self, min_recall: float = 0.8) -> Dict:
        """
        Find threshold that ensures minimum Recall.
        
        Args:
            min_recall: Minimum Recall threshold to achieve.
            
        Returns:
            Dictionary containing threshold and metrics.
        """
        valid_results = []
        
        for threshold in self.thresholds:
            metrics = self._compute_metrics_at_threshold(threshold)
            if metrics['recall'] >= min_recall:
                valid_results.append(metrics)
        
        if not valid_results:
            return self._compute_metrics_at_threshold(0.1)
            
        best_result = max(valid_results, key=lambda x: x['precision'])
        best_result['strategy'] = f'recall_ge_{min_recall}'
        best_result['min_recall_target'] = min_recall
        return best_result
    
    def find_youden_threshold(self) -> Dict:
        """
        Find threshold using Youden's J statistic (Sensitivity + Specificity - 1).
        
        Returns:
            Dictionary containing optimal threshold and metrics.
        """
        best_j = -1
        best_result = None
        
        for threshold in self.thresholds:
            metrics = self._compute_metrics_at_threshold(threshold)
            j_statistic = metrics['recall'] + metrics['specificity'] - 1
            
            if j_statistic > best_j:
                best_j = j_statistic
                best_result = metrics
                best_result['youden_j'] = j_statistic
                
        best_result['strategy'] = 'youden_j'
        return best_result
    
    def get_threshold_analysis_table(self) -> pd.DataFrame:
        """
        Create threshold analysis table with multiple values.
        
        Returns:
            DataFrame containing threshold analysis.
        """
        analysis_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        
        for threshold in analysis_thresholds:
            metrics = self._compute_metrics_at_threshold(threshold)
            results.append(metrics)
            
        return pd.DataFrame(results)
    
    def get_all_optimal_thresholds(self) -> pd.DataFrame:
        """
        Summarize all optimal threshold finding strategies.
        
        Returns:
            DataFrame comparing strategies.
        """
        results = []
        
        f1_result = self.find_optimal_f1()
        results.append({
            'Strategy': 'Max F1',
            'Threshold': f1_result['threshold'],
            'Precision': f1_result['precision'],
            'Recall': f1_result['recall'],
            'F1': f1_result['f1'],
            'Specificity': f1_result['specificity']
        })
        
        recall_80 = self.find_threshold_for_recall(0.8)
        results.append({
            'Strategy': 'Recall >= 80%',
            'Threshold': recall_80['threshold'],
            'Precision': recall_80['precision'],
            'Recall': recall_80['recall'],
            'F1': recall_80['f1'],
            'Specificity': recall_80['specificity']
        })
        
        recall_90 = self.find_threshold_for_recall(0.9)
        results.append({
            'Strategy': 'Recall >= 90%',
            'Threshold': recall_90['threshold'],
            'Precision': recall_90['precision'],
            'Recall': recall_90['recall'],
            'F1': recall_90['f1'],
            'Specificity': recall_90['specificity']
        })
        
        youden = self.find_youden_threshold()
        results.append({
            'Strategy': 'Youden J',
            'Threshold': youden['threshold'],
            'Precision': youden['precision'],
            'Recall': youden['recall'],
            'F1': youden['f1'],
            'Specificity': youden['specificity']
        })
        
        default = self._compute_metrics_at_threshold(0.5)
        results.append({
            'Strategy': 'Default (0.5)',
            'Threshold': 0.5,
            'Precision': default['precision'],
            'Recall': default['recall'],
            'F1': default['f1'],
            'Specificity': default['specificity']
        })
        
        return pd.DataFrame(results)


class CostSensitiveOptimizer:
    """
    Cost-sensitive threshold optimization.
    
    Allows determining optimal threshold when there are different costs for
    False Positive and False Negative.
    """
    
    def __init__(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Initialize CostSensitiveOptimizer.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
        """
        self.y_true = np.array(y_true)
        self.y_proba = np.array(y_proba)
        self.thresholds = np.linspace(0.01, 0.99, 99)
        
    def find_optimal_threshold(
        self,
        cost_fp: float = 1.0,
        cost_fn: float = 1.0,
        cost_tp: float = 0.0,
        cost_tn: float = 0.0
    ) -> Dict:
        """
        Find threshold that minimizes total cost.
        
        Args:
            cost_fp: Cost for False Positive.
            cost_fn: Cost for False Negative.
            cost_tp: Cost for True Positive (usually 0 or negative).
            cost_tn: Cost for True Negative (usually 0 or negative).
            
        Returns:
            Dictionary containing optimal threshold and cost.
        """
        best_cost = float('inf')
        best_result = None
        
        for threshold in self.thresholds:
            y_pred = (self.y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred, labels=[0, 1]).ravel()
            
            total_cost = (cost_fp * fp + cost_fn * fn + 
                         cost_tp * tp + cost_tn * tn)
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_result = {
                    'threshold': threshold,
                    'total_cost': total_cost,
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'fn': fn,
                    'cost_fp': cost_fp,
                    'cost_fn': cost_fn
                }
                
        precision = best_result['tp'] / (best_result['tp'] + best_result['fp']) \
                    if (best_result['tp'] + best_result['fp']) > 0 else 0
        recall = best_result['tp'] / (best_result['tp'] + best_result['fn']) \
                 if (best_result['tp'] + best_result['fn']) > 0 else 0
        
        best_result['precision'] = precision
        best_result['recall'] = recall
        best_result['f1'] = 2 * precision * recall / (precision + recall) \
                           if (precision + recall) > 0 else 0
        
        return best_result
    
    def analyze_cost_scenarios(self) -> pd.DataFrame:
        """
        Analyze multiple cost scenarios.
        
        Returns:
            DataFrame containing analysis results.
        """
        scenarios = [
            {'name': 'Equal Cost', 'cost_fp': 1, 'cost_fn': 1},
            {'name': 'FN 2x costly', 'cost_fp': 1, 'cost_fn': 2},
            {'name': 'FN 5x costly', 'cost_fp': 1, 'cost_fn': 5},
            {'name': 'FP 2x costly', 'cost_fp': 2, 'cost_fn': 1},
            {'name': 'FP 5x costly', 'cost_fp': 5, 'cost_fn': 1},
        ]
        
        results = []
        for scenario in scenarios:
            result = self.find_optimal_threshold(
                cost_fp=scenario['cost_fp'],
                cost_fn=scenario['cost_fn']
            )
            results.append({
                'Scenario': scenario['name'],
                'Cost FP': scenario['cost_fp'],
                'Cost FN': scenario['cost_fn'],
                'Optimal Threshold': result['threshold'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1': result['f1'],
                'Total Cost': result['total_cost']
            })
            
        return pd.DataFrame(results)
