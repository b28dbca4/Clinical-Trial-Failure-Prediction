"""
Feature Engineering Module for Clinical Trial Failure Prediction Project.

This module contains functions to create features from raw data, following the principle:
- Train: fit + transform
- Validation/Test: only transform (using parameters from train)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
import joblib
import json

class FeatureEngineer:
    """
    Class that performs Feature Engineering for clinical trial data.
    
    Attributes:
        params (dict): Stores all parameters fitted from the train set
        scaler (StandardScaler): Feature scaler
        feature_names (list): List of feature names
    """
    
    def __init__(self):
        """Initialize FeatureEngineer with empty parameters."""
        self.params = {}
        self.scaler = None
        self.feature_names = []
        self._is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit parameters on the train set and transform the data.
        
        Args:
            df: DataFrame containing train data
            
        Returns:
            DataFrame with created features
        """
        self._is_fitted = True
        return self._create_all_features(df, fit=True)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            df: DataFrame containing validation/test data
            
        Returns:
            DataFrame with created features
            
        Raises:
            ValueError: If fit_transform has not been called before
        """
        if not self._is_fitted:
            raise ValueError("Phải gọi fit_transform() trước khi gọi transform()")
        return self._create_all_features(df, fit=False)
    
    def _create_all_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Create all features.
        
        Args:
            df: Input DataFrame
            fit: True if fitting (train), False if only transform
            
        Returns:
            DataFrame with new features
        """
        result = df.copy()
        
        result = self._create_temporal_features(result, fit)
        result = self._create_design_features(result, fit)
        result = self._create_complexity_features(result, fit)
        result = self._create_scale_features(result, fit)
        result = self._create_sponsor_features(result, fit)
        result = self._create_intervention_features(result, fit)
        result = self._create_phase_features(result, fit)
        result = self._create_quality_features(result, fit)
        
        return result
    
    def _create_temporal_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Create temporal features.
        
        Features created:
        - study_duration_months: Study duration (months)
        - study_duration_log: Log of study duration
        - start_year: Start year
        - start_quarter: Start quarter
        - is_recent_trial: Recent trial (after 2015)
        - era_category: Era category
        """
        result = df.copy()
        
        # Thời gian nghiên cứu
        if 'start_date' in df.columns and 'completion_date' in df.columns:
            result['study_duration_months'] = (
                (result['completion_date'] - result['start_date']).dt.days / 30.44
            ).clip(lower=0)
            
            if fit:
                self.params['median_duration'] = result['study_duration_months'].median()
            
            result['study_duration_months'] = result['study_duration_months'].fillna(
                self.params['median_duration']
            )
            result['study_duration_log'] = np.log1p(result['study_duration_months'])
        
        # Năm và quý bắt đầu
        if 'start_date' in df.columns:
            result['start_year'] = result['start_date'].dt.year
            result['start_quarter'] = result['start_date'].dt.quarter
            
            if fit:
                self.params['median_year'] = result['start_year'].median()
            
            result['start_year'] = result['start_year'].fillna(self.params['median_year'])
            result['start_quarter'] = result['start_quarter'].fillna(2)
            
            # Nghiên cứu gần đây
            result['is_recent_trial'] = (result['start_year'] >= 2015).astype(int)
            
            # Phân loại thời kỳ
            result['era_category'] = 0
            result.loc[(result['start_year'] >= 2000) & (result['start_year'] < 2010), 'era_category'] = 1
            result.loc[(result['start_year'] >= 2010) & (result['start_year'] < 2020), 'era_category'] = 2
            result.loc[result['start_year'] >= 2020, 'era_category'] = 3
        
        return result
    
    def _create_design_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Create study design features.
        
        Features created:
        - is_randomized: Randomized study
        - is_blinded: Blinded study
        - blinding_level: Blinding level (0-4)
        - is_single_arm: Single-arm study
        - is_parallel: Parallel design
        - is_treatment_purpose: Treatment purpose
        - intervention_model_encoded: Encoded intervention model
        """
        result = df.copy()
        
        if fit:
            self.params['blinding_map'] = {
                'NONE': 0, 'UNKNOWN': 0, 'SINGLE': 1,
                'DOUBLE': 2, 'TRIPLE': 3, 'QUADRUPLE': 4
            }
            self.params['intervention_model_map'] = {
                'SINGLE_GROUP': 0, 'PARALLEL': 1, 'CROSSOVER': 2,
                'FACTORIAL': 3, 'SEQUENTIAL': 4, 'UNKNOWN': 0
            }
        
        if 'allocation' in df.columns:
            result['is_randomized'] = (result['allocation'] == 'RANDOMIZED').astype(int)
        
        if 'masking' in df.columns:
            result['blinding_level'] = result['masking'].map(
                self.params['blinding_map']
            ).fillna(0).astype(int)
            result['is_blinded'] = (result['blinding_level'] > 0).astype(int)
        
        if 'intervention_model' in df.columns:
            result['is_single_arm'] = (result['intervention_model'] == 'SINGLE_GROUP').astype(int)
            result['is_parallel'] = (result['intervention_model'] == 'PARALLEL').astype(int)
            result['intervention_model_encoded'] = result['intervention_model'].map(
                self.params['intervention_model_map']
            ).fillna(0).astype(int)
        
        if 'primary_purpose' in df.columns:
            result['is_treatment_purpose'] = (result['primary_purpose'] == 'TREATMENT').astype(int)
        
        return result
    
    def _create_complexity_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Create complexity features.
        
        Features created:
        - condition_complexity: Condition complexity
        - intervention_complexity: Intervention complexity
        - is_combination_therapy: Combination therapy
        - is_basket_trial: Basket trial
        - total_complexity_score: Total complexity score
        - complexity_score_log: Log complexity score
        """
        result = df.copy()
        
        if 'num_conditions' in df.columns:
            result['condition_complexity'] = pd.cut(
                result['num_conditions'],
                bins=[0, 1, 3, 5, float('inf')],
                labels=[0, 1, 2, 3]
            ).astype(int)
            result['is_basket_trial'] = (result['num_conditions'] >= 3).astype(int)
        
        if 'num_interventions' in df.columns:
            result['intervention_complexity'] = pd.cut(
                result['num_interventions'],
                bins=[0, 1, 2, 4, float('inf')],
                labels=[0, 1, 2, 3]
            ).astype(int)
            result['is_combination_therapy'] = (result['num_interventions'] >= 2).astype(int)
        
        if 'num_conditions' in df.columns and 'num_interventions' in df.columns:
            result['total_complexity_score'] = (
                result['num_conditions'] + result['num_interventions'] * 2
            )
            result['complexity_score_log'] = np.log1p(result['total_complexity_score'])
            
            if fit:
                self.params['complexity_median'] = result['total_complexity_score'].median()
                self.params['complexity_q95'] = result['total_complexity_score'].quantile(0.95)
        
        return result
    
    def _create_scale_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Create scale features.
        
        Features created:
        - enrollment_log: Log enrollment count
        - estimated_arms: Estimated arms
        - enrollment_per_arm: Enrollment per arm
        - enrollment_met: Enrollment met
        """
        result = df.copy()
        
        if 'enrollment_count' in df.columns:
            if fit:
                self.params['enrollment_median'] = result['enrollment_count'].median()
                self.params['enrollment_q99'] = result['enrollment_count'].quantile(0.99)
            
            result['enrollment_count'] = result['enrollment_count'].fillna(
                self.params['enrollment_median']
            )
            
            result['enrollment_count'] = result['enrollment_count'].clip(lower=0, upper=self.params['enrollment_q99'])

            result['enrollment_log'] = np.log1p(result['enrollment_count'])
        
        if 'num_arms' in df.columns:
            result['estimated_arms'] = result['num_arms'].fillna(1).clip(lower=1)
        
        if 'enrollment_count' in result.columns and 'estimated_arms' in result.columns:
            result['enrollment_per_arm'] = (
                result['enrollment_count'] / result['estimated_arms']
            ).clip(lower=1)
        
        if 'enrollment_type' in df.columns:
            result['enrollment_met'] = (result['enrollment_type'] == 'ACTUAL').astype(int)
        
        return result
    
    def _create_sponsor_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Create sponsor features.
        
        Features created:
        - is_industry_sponsor: Industry sponsor
        - sponsor_class_encoded: Encoded sponsor class
        """
        result = df.copy()
        
        if fit:
            self.params['sponsor_mapping'] = {
                'INDUSTRY': 0, 'OTHER': 1, 'NIH': 2,
                'FED': 3, 'NETWORK': 4, 'UNKNOWN': 1
            }
        
        if 'lead_sponsor_class' in df.columns:
            result['is_industry_sponsor'] = (
                result['lead_sponsor_class'] == 'INDUSTRY'
            ).astype(int)
            result['sponsor_class_encoded'] = result['lead_sponsor_class'].map(
                self.params['sponsor_mapping']
            ).fillna(1).astype(int)
        
        return result
    
    def _create_intervention_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Create intervention features.
        
        Features created:
        - has_drug: Has drug
        - has_biological: Has biological
        - has_radiation: Has radiation
        - has_device: Has device
        """
        result = df.copy()
        
        if 'intervention_types' in df.columns:
            intervention_flags = [
                ('has_drug', 'DRUG'),
                ('has_biological', 'BIOLOGICAL'),
                ('has_radiation', 'RADIATION'),
                ('has_device', 'DEVICE')
            ]
            
            for col_name, int_type in intervention_flags:
                result[col_name] = result['intervention_types'].str.contains(
                    int_type, case=False, na=False
                ).astype(int)
        
        return result
    
    def _create_phase_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Create phase features.
        
        Features created:
        - phase_encoded: Encoded phase (0-4)
        - is_late_phase: Late phase (Phase 3-4)
        """
        result = df.copy()
        
        if fit:
            self.params['phase_mapping'] = {
                'EARLY_PHASE1': 0, 'PHASE1': 1, 'PHASE1|PHASE2': 1.5,
                'PHASE2': 2, 'PHASE2|PHASE3': 2.5, 'PHASE3': 3, 'PHASE4': 4, 'NA': 1
            }
        
        if 'phases' in df.columns:
            def encode_phase(phase_str):
                if pd.isna(phase_str) or phase_str == 'NA':
                    return 1
                for key, val in self.params['phase_mapping'].items():
                    if key in str(phase_str):
                        return val
                return 1
            
            result['phase_encoded'] = result['phases'].apply(encode_phase)
            result['is_late_phase'] = (result['phase_encoded'] >= 3).astype(int)
        
        return result
    
    def _create_quality_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Create quality features.
        
        Features created:
        - has_dmc: Has DMC
        - has_results: Has results
        """
        result = df.copy()
        
        if 'has_dmc' in df.columns:
            result['has_dmc'] = result['has_dmc'].map(
                {True: 1, False: 0, 'True': 1, 'False': 0, 'UNKNOWN': 0}
            ).fillna(0).astype(int)
        
        if 'has_results' in df.columns:
            result['has_results'] = result['has_results'].map(
                {True: 1, False: 0, 'True': 1, 'False': 0}
            ).fillna(0).astype(int)
        
        return result
    
    def get_feature_list(self) -> List[str]:
        """
        Get the list of numerical features for the model.
        
        Returns:
            List of feature names
        """
        return [
            # Temporal
            'study_duration_months', 'study_duration_log', 'start_year', 
            'start_quarter', 'is_recent_trial', 'era_category',
            # Design
            'is_randomized', 'is_blinded', 'blinding_level', 'is_single_arm',
            'is_parallel', 'is_treatment_purpose', 'intervention_model_encoded',
            # Complexity
            'num_conditions', 'condition_complexity', 'num_interventions',
            'intervention_complexity', 'is_combination_therapy', 'is_basket_trial',
            'total_complexity_score', 'complexity_score_log',
            # Scale
            'enrollment_count', 'enrollment_log', 'estimated_arms', 'enrollment_per_arm',
            # Sponsor
            'is_industry_sponsor', 'sponsor_class_encoded',
            # Intervention
            'has_drug', 'has_biological', 'has_radiation', 'has_device',
            # Phase
            'phase_encoded', 'is_late_phase',
            # Quality
            'has_dmc', 'has_results', 'enrollment_met'
        ]
    
    def save_params(self, filepath: str) -> None:
        """
        Save fitted parameters to a JSON file.
        
        Args:
            filepath: File path
        """
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        params_serializable = convert_to_serializable(self.params)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params_serializable, f, indent=2, ensure_ascii=False)
    
    def load_params(self, filepath: str) -> None:
        """
        Load parameters from a JSON file.
        
        Args:
            filepath: File path
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        self._is_fitted = True


class FeatureScaler:
    """
    Class for feature scaling using StandardScaler.
    
    Follows the principle:
    - fit on train
    - transform on train/val/test
    """
    
    def __init__(self):
        """Initialize FeatureScaler."""
        self.scaler = StandardScaler()
        self._is_fitted = False
        self.feature_names = []
    
    def fit_transform(self, X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Fit scaler on train data and transform.
        
        Args:
            X: Input DataFrame
            feature_cols: List of feature columns to scale
            
        Returns:
            Scaled DataFrame
        """
        self.feature_names = [col for col in feature_cols if col in X.columns]
        X_features = X[self.feature_names].copy()
        
        # Xử lý missing values
        for col in self.feature_names:
            if X_features[col].isnull().any():
                X_features[col] = X_features[col].fillna(X_features[col].median())
        
        # Fit và transform
        X_scaled = self.scaler.fit_transform(X_features)
        self._is_fitted = True
        
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scaler.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Scaled DataFrame
            
        Raises:
            ValueError: If not fitted
        """
        if not self._is_fitted:
            raise ValueError("Phải gọi fit_transform() trước khi gọi transform()")
        
        X_features = X[self.feature_names].copy()
        
        # Xử lý missing với median từ train (đã được lưu trong scaler)
        for i, col in enumerate(self.feature_names):
            if X_features[col].isnull().any():
                X_features[col] = X_features[col].fillna(self.scaler.mean_[i])
        
        X_scaled = self.scaler.transform(X_features)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
    
    def save(self, filepath: str) -> None:
        """Save scaler to file."""
        joblib.dump(self.scaler, filepath)
    
    def load(self, filepath: str) -> None:
        """Load scaler from file."""
        self.scaler = joblib.load(filepath)
        self._is_fitted = True
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get mean and std statistics of the scaler.
        
        Returns:
            Dictionary containing mean and std of each feature
        """
        if not self._is_fitted:
            return {}
        
        stats = {}
        for i, name in enumerate(self.feature_names):
            stats[name] = {
                'mean': float(self.scaler.mean_[i]),
                'std': float(self.scaler.scale_[i])
            }
        return stats
