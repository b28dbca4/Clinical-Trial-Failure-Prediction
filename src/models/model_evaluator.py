"""
Module đánh giá mô hình cho dự án Clinical Trial Failure Prediction.

Cung cấp các class và hàm tiện ích để đánh giá hiệu suất mô hình,
phân tích lỗi và tạo báo cáo.
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
    Lớp đánh giá và phân tích mô hình phân loại.
    
    Attributes:
        model: Mô hình cần đánh giá.
        feature_names (list): Danh sách tên features.
    """
    
    def __init__(self, model: Any, feature_names: List[str]):
        """
        Khởi tạo ModelEvaluator.
        
        Args:
            model: Mô hình đã huấn luyện.
            feature_names: Danh sách tên các features.
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
        Đánh giá mô hình trên tập dữ liệu.
        
        Args:
            X: Features.
            y: Labels.
            dataset_name: Tên tập dữ liệu.
            
        Returns:
            Dictionary chứa các metrics.
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
        Tạo classification report.
        
        Args:
            X: Features.
            y: Labels.
            target_names: Tên các class.
            
        Returns:
            Classification report dạng string.
        """
        if target_names is None:
            target_names = ['That bai (0)', 'Thanh cong (1)']
        
        y_pred = self.model.predict(X)
        return classification_report(y, y_pred, target_names=target_names)
    
    def get_confusion_matrix(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, Dict]:
        """
        Tính confusion matrix và các thành phần.
        
        Args:
            X: Features.
            y: Labels.
            
        Returns:
            Tuple gồm (confusion matrix, dict các thành phần).
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
        Lấy feature importance từ mô hình.
        
        Returns:
            DataFrame chứa feature importance.
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Mo hinh khong ho tro feature importance")
        
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
        Phân tích các trường hợp dự đoán sai.
        
        Args:
            X: Features.
            y: Labels.
            ids: ID của các samples.
            
        Returns:
            DataFrame chứa thông tin phân tích lỗi.
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
        Tổng hợp phân tích lỗi.
        
        Args:
            error_df: DataFrame từ analyze_errors.
            
        Returns:
            DataFrame tổng hợp theo loại lỗi.
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
        So sánh đặc điểm giữa các nhóm lỗi.
        
        Args:
            error_df: DataFrame từ analyze_errors.
            features: Danh sách features cần so sánh.
            
        Returns:
            DataFrame so sánh mean của các features.
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
        Vẽ confusion matrix.
        
        Args:
            X: Features.
            y: Labels.
            ax: Matplotlib axes.
            title: Tiêu đề biểu đồ.
            
        Returns:
            Matplotlib axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        cm, _ = self.get_confusion_matrix(X, y)
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['That bai (0)', 'Thanh cong (1)'],
            yticklabels=['That bai (0)', 'Thanh cong (1)']
        )
        ax.set_xlabel('Du doan')
        ax.set_ylabel('Thuc te')
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
        Vẽ đường cong ROC.
        
        Args:
            X: Features.
            y: Labels.
            ax: Matplotlib axes.
            title: Tiêu đề biểu đồ.
            
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
        Vẽ đường cong Precision-Recall.
        
        Args:
            X: Features.
            y: Labels.
            ax: Matplotlib axes.
            title: Tiêu đề biểu đồ.
            
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
        Vẽ biểu đồ feature importance.
        
        Args:
            top_n: Số features hiển thị.
            ax: Matplotlib axes.
            title: Tiêu đề biểu đồ.
            
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
        Tạo bảng tổng hợp so sánh nhiều mô hình.
        
        Args:
            models_dict: Dictionary chứa các mô hình.
            X_test: Features test set.
            y_test: Labels test set.
            
        Returns:
            DataFrame so sánh các mô hình.
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
