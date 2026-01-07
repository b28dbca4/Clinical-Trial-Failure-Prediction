"""
Model evaluation module for Clinical Trial Failure Prediction project.

Provides classes and utility functions for evaluating model performance,
error analysis and report generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)


class ModelEvaluator:
    """
    Class for evaluating and analyzing classification models.
    
    Attributes:
        model: Model to evaluate.
        feature_names (list): List of feature names.
    """
    
    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initialize ModelEvaluator.
        
        Args:
            model: Trained model.
            feature_names: List of feature names.
        """
        self.model = model
        self.feature_names = feature_names
        self.metrics = {}
        self.predictions = {}
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = 'test'
    ) -> Dict:
        """
        Evaluate model on dataset.
        
        Args:
            X: Features.
            y: Labels.
            dataset_name: Name of dataset.
            
        Returns:
            Dictionary containing metrics.
        """
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba),
            'average_precision': average_precision_score(y, y_proba)
        }
        
        self.metrics[dataset_name] = metrics
        self.predictions[dataset_name] = {
            'y_true': y,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return metrics
    
    def get_classification_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_names: List[str] = None
    ) -> str:
        """
        Generate classification report.
        
        Args:
            X: Features.
            y: Labels.
            target_names: Names of classes.
            
        Returns:
            Classification report as string.
        """
        if target_names is None:
            target_names = ['Failed (0)', 'Success (1)']
        
        y_pred = self.model.predict(X)
        return classification_report(y, y_pred, target_names=target_names)
    
    def get_confusion_matrix(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, Dict]:
        """
        Calculate confusion matrix and its components.
        
        Args:
            X: Features.
            y: Labels.
            
        Returns:
            Tuple of (confusion matrix, dict of components).
        """
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        components = {
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
        
        return cm, components
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from model.
        
        Returns:
            DataFrame containing feature importance.
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def analyze_errors(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ids: pd.Series = None
    ) -> pd.DataFrame:
        """
        Analyze misclassified cases.
        
        Args:
            X: Features.
            y: Labels.
            ids: IDs of samples.
            
        Returns:
            DataFrame containing error analysis information.
        """
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        error_df = X.copy()
        error_df['y_true'] = y.values
        error_df['y_pred'] = y_pred
        error_df['y_proba'] = y_proba
        
        if ids is not None:
            error_df['id'] = ids.values
        
        error_df['error_type'] = 'Correct'
        error_df.loc[
            (error_df['y_true'] == 0) & (error_df['y_pred'] == 1),
            'error_type'
        ] = 'False Positive'
        error_df.loc[
            (error_df['y_true'] == 1) & (error_df['y_pred'] == 0),
            'error_type'
        ] = 'False Negative'
        
        return error_df
    
    def get_error_summary(self, error_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize error analysis.
        
        Args:
            error_df: DataFrame from analyze_errors.
            
        Returns:
            DataFrame summarized by error type.
        """
        summary = error_df.groupby('error_type').agg({
            'y_true': 'count'
        }).rename(columns={'y_true': 'count'})
        
        summary['percentage'] = summary['count'] / len(error_df) * 100
        
        return summary
    
    def compare_error_groups(
        self,
        error_df: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Compare characteristics between error groups.
        
        Args:
            error_df: DataFrame from analyze_errors.
            features: List of features to compare.
            
        Returns:
            DataFrame comparing mean of features.
        """
        comparison = error_df.groupby('error_type')[features].mean()
        return comparison
    
    def plot_confusion_matrix(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ax: plt.Axes = None,
        title: str = 'Confusion Matrix'
    ) -> plt.Axes:
        """
        Plot confusion matrix.
        
        Args:
            X: Features.
            y: Labels.
            ax: Matplotlib axes.
            title: Plot title.
            
        Returns:
            Matplotlib axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        cm, _ = self.get_confusion_matrix(X, y)
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Failed (0)', 'Success (1)'],
            yticklabels=['Failed (0)', 'Success (1)']
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        
        return ax
    
    def plot_roc_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ax: plt.Axes = None,
        title: str = 'ROC Curve'
    ) -> plt.Axes:
        """
        Plot ROC curve.
        
        Args:
            X: Features.
            y: Labels.
            ax: Matplotlib axes.
            title: Plot title.
            
        Returns:
            Matplotlib axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        y_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        
        ax.plot(fpr, tpr, color='blue', linewidth=2,
                label=f'ROC Curve (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
        ax.fill_between(fpr, tpr, alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')
        
        return ax
    
    def plot_precision_recall_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ax: plt.Axes = None,
        title: str = 'Precision-Recall Curve'
    ) -> plt.Axes:
        """
        Plot Precision-Recall curve.
        
        Args:
            X: Features.
            y: Labels.
            ax: Matplotlib axes.
            title: Plot title.
            
        Returns:
            Matplotlib axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        y_proba = self.model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_proba)
        ap = average_precision_score(y, y_proba)
        
        ax.plot(recall, precision, color='green', linewidth=2,
                label=f'PR Curve (AP = {ap:.4f})')
        ax.axhline(y=y.mean(), color='red', linestyle='--',
                   label=f'Baseline ({y.mean():.4f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc='upper right')
        
        return ax
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        ax: plt.Axes = None,
        title: str = 'Feature Importance'
    ) -> plt.Axes:
        """
        Plot feature importance chart.
        
        Args:
            top_n: Number of features to display.
            ax: Matplotlib axes.
            title: Plot title.
            
        Returns:
            Matplotlib axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        importance_df = self.get_feature_importance().head(top_n)
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance_df)))
        ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.invert_yaxis()
        
        return ax
    
    def generate_summary_table(
        self,
        models_dict: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Generate summary table comparing multiple models.
        
        Args:
            models_dict: Dictionary containing models.
            X_test: Test set features.
            y_test: Test set labels.
            
        Returns:
            DataFrame comparing models.
        """
        data = []
        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            data.append({
                'Model': name,
                'ROC-AUC': roc_auc_score(y_test, y_proba),
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('ROC-AUC', ascending=False)
        return df.round(4)
