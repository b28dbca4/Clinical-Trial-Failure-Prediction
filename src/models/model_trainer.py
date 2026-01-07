"""
Model training module for Clinical Trial Failure Prediction project.

Provides classes and utility functions for training classification models
including Logistic Regression, Random Forest, XGBoost and LightGBM.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score

import xgboost as xgb
import lightgbm as lgb


class ModelTrainer:
    """
    Class for training and managing classification models.
    
    Attributes:
        seed (int): Random seed value to ensure reproducibility.
        models (dict): Dictionary storing trained models.
        results (dict): Dictionary storing evaluation results.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            seed: Random seed value.
        """
        self.seed = seed
        self.models = {}
        self.results = {}
        np.random.seed(seed)
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **kwargs
    ) -> LogisticRegression:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training data.
            y_train: Training labels.
            X_val: Validation data.
            y_val: Validation labels.
            **kwargs: Additional parameters for the model.
            
        Returns:
            Trained LogisticRegression model.
        """
        default_params = {
            'random_state': self.seed,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'class_weight': 'balanced'
        }
        default_params.update(kwargs)
        
        model = LogisticRegression(**default_params)
        model.fit(X_train, y_train)
        
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = model.predict(X_val)
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = {
            'val_auc': roc_auc_score(y_val, y_val_proba),
            'val_f1': f1_score(y_val, y_val_pred)
        }
        
        return model
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training data.
            y_train: Training labels.
            X_val: Validation data.
            y_val: Validation labels.
            **kwargs: Additional parameters for the model.
            
        Returns:
            Trained RandomForestClassifier model.
        """
        default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': self.seed,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = model.predict(X_val)
        
        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'val_auc': roc_auc_score(y_val, y_val_proba),
            'val_f1': f1_score(y_val, y_val_pred)
        }
        
        return model
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        early_stopping_rounds: int = 50,
        **kwargs
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model with Early Stopping.
        
        Args:
            X_train: Training data.
            y_train: Training labels.
            X_val: Validation data.
            y_val: Validation labels.
            early_stopping_rounds: Number of early stopping rounds.
            **kwargs: Additional parameters for the model.
            
        Returns:
            Trained XGBClassifier model.
        """
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.seed,
            'use_label_encoder': False,
            'eval_metric': 'auc',
            'early_stopping_rounds': early_stopping_rounds
        }
        default_params.update(kwargs)
        
        model = xgb.XGBClassifier(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = model.predict(X_val)
        
        self.models['xgboost'] = model
        self.results['xgboost'] = {
            'val_auc': roc_auc_score(y_val, y_val_proba),
            'val_f1': f1_score(y_val, y_val_pred),
            'best_iteration': model.best_iteration
        }
        
        return model
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        early_stopping_rounds: int = 50,
        **kwargs
    ) -> lgb.LGBMClassifier:
        """
        Train LightGBM model with Early Stopping.
        
        Args:
            X_train: Training data.
            y_train: Training labels.
            X_val: Validation data.
            y_val: Validation labels.
            early_stopping_rounds: Number of early stopping rounds.
            **kwargs: Additional parameters for the model.
            
        Returns:
            Trained LGBMClassifier model.
        """
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.seed,
            'verbose': -1
        }
        default_params.update(kwargs)
        
        model = lgb.LGBMClassifier(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        )
        
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = model.predict(X_val)
        
        self.models['lightgbm'] = model
        self.results['lightgbm'] = {
            'val_auc': roc_auc_score(y_val, y_val_proba),
            'val_f1': f1_score(y_val, y_val_pred),
            'best_iteration': model.best_iteration_
        }
        
        return model
    
    def tune_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        param_grid: Optional[Dict] = None
    ) -> xgb.XGBClassifier:
        """
        Tune XGBoost parameters using GridSearchCV with Predefined Split.
        
        Args:
            X_train: Training data.
            y_train: Training labels.
            X_val: Validation data.
            y_val: Validation labels.
            param_grid: Grid of parameters to search.
            
        Returns:
            Optimized XGBClassifier model.
        """
        if param_grid is None:
            param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [100, 200],
                'subsample': [0.8],
                'colsample_bytree': [0.8]
            }
        
        X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
        y_combined = pd.concat([y_train, y_val], axis=0, ignore_index=True)
        
        split_index = [-1] * len(X_train) + [0] * len(X_val)
        ps = PredefinedSplit(test_fold=split_index)
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        base_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            use_label_encoder=False,
            eval_metric='auc'
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=ps,
            scoring='roc_auc',
            n_jobs=1,
            verbose=1
        )
        
        grid_search.fit(X_combined, y_combined)
        
        best_model = grid_search.best_estimator_
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        y_val_pred = best_model.predict(X_val)
        
        self.models['xgboost_tuned'] = best_model
        self.results['xgboost_tuned'] = {
            'val_auc': roc_auc_score(y_val, y_val_proba),
            'val_f1': f1_score(y_val, y_val_pred),
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }
        
        return best_model
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table of models.
        
        Returns:
            DataFrame containing comparison information.
        """
        data = []
        for name, result in self.results.items():
            data.append({
                'Model': name,
                'Val ROC-AUC': result['val_auc'],
                'Val F1-Score': result['val_f1']
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Val ROC-AUC', ascending=False)
        return df
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best model by ROC-AUC score.
        
        Returns:
            Tuple of (model name, model object).
        """
        best_name = max(self.results, key=lambda x: self.results[x]['val_auc'])
        return best_name, self.models[best_name]
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        feature_columns: List[str],
        metrics: Dict,
        save_path: Path
    ) -> str:
        """
        Save model and metadata.
        
        Args:
            model: Model to save.
            model_name: Model name.
            feature_columns: List of feature names.
            metrics: Dictionary of metrics.
            save_path: Directory path to save to.
            
        Returns:
            Path of saved file.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'clinical_trial_model_{timestamp}.joblib'
        filepath = save_path / filename
        
        model_package = {
            'model': model,
            'model_name': model_name,
            'feature_columns': feature_columns,
            'metrics': metrics,
            'training_date': timestamp
        }
        
        joblib.dump(model_package, filepath)
        
        fixed_path = save_path / 'best_model.joblib'
        joblib.dump(model_package, fixed_path)
        
        return str(filepath)
