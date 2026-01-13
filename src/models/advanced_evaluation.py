"""
Module danh gia mo hinh nang cao (Advanced Model Evaluation).

Cung cap cac cong cu danh gia toan dien cho bai toan phan lop nhi phan
voi du lieu mat can bang:
- Tinh toan nhieu metric dong thoi
- Cross-validation stability analysis
- Bootstrap confidence intervals
- Model comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import StratifiedKFold
import warnings


class AdvancedModelEvaluator:
    """
    Lop danh gia mo hinh nang cao cho bai toan phan lop nhi phan.
    
    Attributes:
        pos_label: Nhan cua lop duong (positive class).
    """
    
    def __init__(self, pos_label: int = 1):
        """
        Khoi tao AdvancedModelEvaluator.
        
        Args:
            pos_label: Nhan cua lop duong.
        """
        self.pos_label = pos_label
        
    def compute_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """
        Tinh toan bo metric day du cho bai toan mat can bang.
        
        Args:
            y_true: Nhan thuc te.
            y_pred: Du doan nhi phan.
            y_proba: Xac suat du doan lop duong.
            
        Returns:
            Dictionary chua tat ca cac metric.
        """
        metrics = {}
        
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        metrics['precision_pos'] = precision_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        metrics['recall_pos'] = recall_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        metrics['f1_pos'] = f1_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        
        neg_label = 1 - self.pos_label
        metrics['precision_neg'] = precision_score(y_true, y_pred, pos_label=neg_label, zero_division=0)
        metrics['recall_neg'] = recall_score(y_true, y_pred, pos_label=neg_label, zero_division=0)
        metrics['f1_neg'] = f1_score(y_true, y_pred, pos_label=neg_label, zero_division=0)
        
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        metrics['tp'] = tp
        metrics['fp'] = fp
        metrics['tn'] = tn
        metrics['fn'] = fn
        
        return metrics
    
    def compute_metrics_at_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Tinh metric tai mot nguong cu the.
        
        Args:
            y_true: Nhan thuc te.
            y_proba: Xac suat du doan.
            threshold: Nguong quyet dinh.
            
        Returns:
            Dictionary chua cac metric.
        """
        y_pred = (y_proba >= threshold).astype(int)
        metrics = self.compute_comprehensive_metrics(y_true, y_pred, y_proba)
        metrics['threshold'] = threshold
        return metrics
    
    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict:
        """
        Thuc hien Stratified K-Fold Cross-Validation.
        
        Args:
            model: Mo hinh (phai co fit va predict_proba).
            X: Features.
            y: Labels.
            n_splits: So fold.
            random_state: Random seed.
            
        Returns:
            Dictionary chua ket qua CV (mean, std cua cac metric).
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        cv_results = {
            'roc_auc': [],
            'pr_auc': [],
            'f1_pos': [],
            'recall_pos': [],
            'precision_pos': [],
            'mcc': [],
            'balanced_accuracy': []
        }
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            model_clone = model.__class__(**model.get_params())
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_clone.fit(X_train_cv, y_train_cv)
            
            y_proba_cv = model_clone.predict_proba(X_val_cv)[:, 1]
            y_pred_cv = model_clone.predict(X_val_cv)
            
            cv_results['roc_auc'].append(roc_auc_score(y_val_cv, y_proba_cv))
            cv_results['pr_auc'].append(average_precision_score(y_val_cv, y_proba_cv))
            cv_results['f1_pos'].append(f1_score(y_val_cv, y_pred_cv, pos_label=self.pos_label, zero_division=0))
            cv_results['recall_pos'].append(recall_score(y_val_cv, y_pred_cv, pos_label=self.pos_label, zero_division=0))
            cv_results['precision_pos'].append(precision_score(y_val_cv, y_pred_cv, pos_label=self.pos_label, zero_division=0))
            cv_results['mcc'].append(matthews_corrcoef(y_val_cv, y_pred_cv))
            cv_results['balanced_accuracy'].append(balanced_accuracy_score(y_val_cv, y_pred_cv))
        
        summary = {}
        for metric, values in cv_results.items():
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_values'] = values
            
        return summary
    
    def bootstrap_confidence_interval(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric_func: callable,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 42
    ) -> Dict:
        """
        Tinh khoang tin cay bang Bootstrap.
        
        Args:
            y_true: Nhan thuc te.
            y_proba: Xac suat du doan.
            metric_func: Ham tinh metric (nhan y_true, y_proba).
            n_bootstrap: So lan bootstrap.
            confidence_level: Muc tin cay (0.95 = 95%).
            random_state: Random seed.
            
        Returns:
            Dictionary chua point estimate va confidence interval.
        """
        np.random.seed(random_state)
        n_samples = len(y_true)
        
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_proba_boot = y_proba[indices]
            
            if len(np.unique(y_true_boot)) < 2:
                continue
                
            try:
                score = metric_func(y_true_boot, y_proba_boot)
                bootstrap_scores.append(score)
            except:
                continue
                
        bootstrap_scores = np.array(bootstrap_scores)
        
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
        
        return {
            'point_estimate': metric_func(y_true, y_proba),
            'ci_lower': lower,
            'ci_upper': upper,
            'confidence_level': confidence_level,
            'std': np.std(bootstrap_scores),
            'n_valid_bootstrap': len(bootstrap_scores)
        }
    
    def get_pr_auc_ci(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Tinh khoang tin cay cho PR-AUC.
        
        Args:
            y_true: Nhan thuc te.
            y_proba: Xac suat du doan.
            n_bootstrap: So lan bootstrap.
            confidence_level: Muc tin cay.
            
        Returns:
            Dictionary chua PR-AUC va CI.
        """
        return self.bootstrap_confidence_interval(
            y_true, y_proba,
            metric_func=average_precision_score,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )
    
    def get_roc_auc_ci(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Tinh khoang tin cay cho ROC-AUC.
        
        Args:
            y_true: Nhan thuc te.
            y_proba: Xac suat du doan.
            n_bootstrap: So lan bootstrap.
            confidence_level: Muc tin cay.
            
        Returns:
            Dictionary chua ROC-AUC va CI.
        """
        return self.bootstrap_confidence_interval(
            y_true, y_proba,
            metric_func=roc_auc_score,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )


def compare_models_comprehensive(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    So sanh toan dien nhieu mo hinh.
    
    Args:
        models_dict: Dictionary {ten_model: model_object}.
        X_test: Test features.
        y_test: Test labels.
        threshold: Nguong quyet dinh.
        
    Returns:
        DataFrame so sanh cac mo hinh.
    """
    evaluator = AdvancedModelEvaluator()
    results = []
    
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = evaluator.compute_comprehensive_metrics(
            y_test.values, y_pred, y_proba
        )
        metrics['model'] = name
        results.append(metrics)
        
    df = pd.DataFrame(results)
    
    column_order = [
        'model', 'roc_auc', 'pr_auc', 'f1_pos', 'recall_pos', 'precision_pos',
        'balanced_accuracy', 'mcc', 'specificity', 'accuracy'
    ]
    available_cols = [c for c in column_order if c in df.columns]
    df = df[available_cols]
    df = df.sort_values('pr_auc', ascending=False)
    
    return df


def create_metrics_heatmap_data(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Tao du lieu cho heatmap so sanh metric giua cac mo hinh.
    
    Args:
        models_dict: Dictionary {ten_model: model_object}.
        X_test: Test features.
        y_test: Test labels.
        threshold: Nguong quyet dinh.
        
    Returns:
        DataFrame voi index la model, columns la metric.
    """
    df = compare_models_comprehensive(models_dict, X_test, y_test, threshold)
    
    metrics_cols = ['roc_auc', 'pr_auc', 'f1_pos', 'recall_pos', 
                    'precision_pos', 'balanced_accuracy', 'mcc']
    
    available_metrics = [c for c in metrics_cols if c in df.columns]
    
    heatmap_df = df.set_index('model')[available_metrics]
    
    heatmap_df.columns = ['ROC-AUC', 'PR-AUC', 'F1(Pos)', 'Recall(Pos)', 
                          'Precision(Pos)', 'Balanced Acc', 'MCC'][:len(available_metrics)]
    
    return heatmap_df


def analyze_class_imbalance_impact(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray
) -> Dict:
    """
    Phan tich tac dong cua mat can bang lop.
    
    Args:
        y_true: Nhan thuc te.
        y_proba: Xac suat du doan.
        y_pred: Du doan nhi phan.
        
    Returns:
        Dictionary chua phan tich.
    """
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    imbalance_ratio = n_neg / n_pos if n_pos > 0 else float('inf')
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    baseline_accuracy = max(n_pos, n_neg) / len(y_true)
    
    improvement_over_baseline = accuracy - baseline_accuracy
    
    return {
        'n_positive': n_pos,
        'n_negative': n_neg,
        'imbalance_ratio': imbalance_ratio,
        'positive_rate': n_pos / len(y_true),
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'baseline_accuracy': baseline_accuracy,
        'improvement_over_baseline': improvement_over_baseline,
        'accuracy_gap': accuracy - balanced_acc,
        'true_positive_rate': tp / n_pos if n_pos > 0 else 0,
        'true_negative_rate': tn / n_neg if n_neg > 0 else 0
    }
