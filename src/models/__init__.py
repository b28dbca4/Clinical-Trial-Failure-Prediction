"""
Models Module - Clinical Trial Failure Prediction

This module provides comprehensive machine learning capabilities:
- 25+ ML algorithms for comparison
- Hyperparameter optimization with Optuna
- Comprehensive model evaluation (30+ metrics)
- Model interpretation (SHAP, LIME, PDP, ICE)
- Cross-validation and validation techniques

Main Components:
- ComprehensiveModelTrainer: Train and compare 25+ models
- ComprehensiveModelEvaluator: Evaluate with 30+ metrics and 12+ visualizations
- ModelInterpreter: Interpret predictions with SHAP, LIME, etc.
"""

from .model_trainer_comprehensive import (
    ComprehensiveModelTrainer,
    train_and_compare_all_models
)

from .model_evaluator_comprehensive import (
    ComprehensiveModelEvaluator,
    evaluate_and_compare_models
)

from .model_interpretation import (
    ModelInterpreter,
    interpret_model
)

__all__ = [
    # Trainer
    'ComprehensiveModelTrainer',
    'train_and_compare_all_models',
    
    # Evaluator
    'ComprehensiveModelEvaluator',
    'evaluate_and_compare_models',
    
    # Interpreter
    'ModelInterpreter',
    'interpret_model',
]

# Version
__version__ = '1.0.0'

# Quick usage examples
USAGE_EXAMPLES = """
QUICK START EXAMPLES:

1. Train and Compare All Models:
   -------------------------------
   from src.models import train_and_compare_all_models
   
   results, best_model = train_and_compare_all_models(
       X_train, y_train, X_test, y_test,
       use_cross_validation=True,
       optimize_top_n=3
   )

2. Evaluate Multiple Models:
   --------------------------
   from src.models import evaluate_and_compare_models
   
   models_dict = {
       'Random Forest': rf_model,
       'XGBoost': xgb_model,
       'LightGBM': lgb_model
   }
   
   evaluator = evaluate_and_compare_models(
       models_dict, X_test, y_test
   )

3. Interpret a Model:
   -------------------
   from src.models import interpret_model
   
   interpreter = interpret_model(
       best_model, X_train, X_test,
       feature_names=feature_names,
       sample_indices=[0, 1, 2]
   )

DETAILED USAGE:

1. Training with ComprehensiveModelTrainer:
   ----------------------------------------
   from src.models import ComprehensiveModelTrainer
   
   trainer = ComprehensiveModelTrainer(
       X_train, y_train, X_test, y_test
   )
   
   # Prepare data with balancing
   trainer.prepare_data(balance_method='smote')
   
   # Train all models
   results = trainer.train_all_models(use_cross_validation=True, cv_folds=5)
   
   # Optimize top models
   best_models = trainer.optimize_top_models(top_n=3, n_trials=100)
   
   # Get results
   results_df = trainer.get_results_dataframe()
   trainer.print_comparison_report()
   
   # Save best model
   trainer.save_best_model('best_model.pkl')

2. Evaluation with ComprehensiveModelEvaluator:
   --------------------------------------------
   from src.models import ComprehensiveModelEvaluator
   
   evaluator = ComprehensiveModelEvaluator()
   
   # Evaluate single model
   metrics = evaluator.evaluate_single_model(
       model, X_test, y_test, 'XGBoost'
   )
   
   # Generate full report
   evaluator.generate_full_report(
       'XGBoost', X_test, feature_names
   )
   
   # Compare all models
   evaluator.plot_roc_curve()
   evaluator.plot_metric_comparison()
   evaluator.compare_all_models()
   
   # Export results
   evaluator.export_results_to_csv()

3. Interpretation with ModelInterpreter:
   -------------------------------------
   from src.models import ModelInterpreter
   
   interpreter = ModelInterpreter(
       model, X_train, X_test, feature_names
   )
   
   # Feature importance
   interpreter.plot_feature_importance(method='shap')
   
   # SHAP analysis
   interpreter.plot_shap_summary()
   interpreter.plot_shap_waterfall(sample_idx=0)
   interpreter.plot_shap_dependence(feature_idx=5)
   
   # Partial dependence
   interpreter.plot_partial_dependence([0, 1, 2])
   interpreter.plot_ice_curves(feature_idx=0)
   
   # LIME explanation
   interpreter.explain_with_lime(sample_idx=0)
   
   # Full report
   interpreter.generate_full_interpretation_report([0, 1, 2])
"""

def print_usage():
    """Print usage examples"""
    print(USAGE_EXAMPLES)


def get_available_models():
    """Get list of all available models"""
    return [
        # Linear Models
        'Logistic Regression',
        'Ridge Classifier',
        'SGD Classifier',
        'Passive Aggressive',
        'Perceptron',
        
        # Tree Models
        'Decision Tree',
        'Random Forest',
        'Extra Trees',
        'Extra Trees (Small)',
        
        # Gradient Boosting
        'XGBoost',
        'LightGBM',
        'CatBoost',
        'Gradient Boosting',
        'Hist Gradient Boosting',
        
        # SVM
        'Linear SVM',
        'RBF SVM',
        'Polynomial SVM',
        
        # Neural Networks
        'Neural Network (Small)',
        'Neural Network (Medium)',
        'Neural Network (Large)',
        
        # Naive Bayes
        'Gaussian Naive Bayes',
        'Bernoulli Naive Bayes',
        'Complement Naive Bayes',
        
        # K-Nearest Neighbors
        'KNN (k=5)',
        'KNN (k=10)',
        
        # Discriminant Analysis
        'Linear Discriminant Analysis',
        'Quadratic Discriminant Analysis',
        
        # Ensemble
        'AdaBoost',
        'Bagging Classifier',
        'Voting Classifier (Soft)'
    ]


def get_evaluation_metrics():
    """Get list of all evaluation metrics"""
    return [
        # Core Classification Metrics
        'accuracy',
        'balanced_accuracy',
        'precision',
        'recall',
        'f1_score',
        'f2_score',
        'f05_score',
        
        # ROC and PR
        'roc_auc',
        'pr_auc',
        'avg_precision',
        
        # Statistical Metrics
        'mcc',  # Matthews Correlation Coefficient
        'cohen_kappa',
        'jaccard',
        
        # Probabilistic Metrics
        'log_loss',
        'brier_score',
        
        # Error Metrics
        'mse',
        'rmse',
        'mae',
        
        # Confusion Matrix Components
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives',
        
        # Derived Metrics
        'specificity',
        'sensitivity',
        'fpr',  # False Positive Rate
        'fnr',  # False Negative Rate
        'npv',  # Negative Predictive Value
        'ppv',  # Positive Predictive Value
        'youdens_j',  # Youden's J Statistic
        'diagnostic_odds_ratio'
    ]


def get_interpretation_methods():
    """Get list of all interpretation methods"""
    return [
        # Feature Importance
        'Built-in Feature Importance',
        'Permutation Importance',
        'SHAP Feature Importance',
        
        # SHAP Analysis
        'SHAP Summary Plot',
        'SHAP Waterfall Plot',
        'SHAP Force Plot',
        'SHAP Dependence Plot',
        
        # Partial Dependence
        'Partial Dependence Plot (PDP)',
        'Individual Conditional Expectation (ICE)',
        
        # Local Explanations
        'LIME Explanation',
        
        # Tree Visualization
        'Decision Tree Visualization',
        'Tree Rules Export'
    ]
