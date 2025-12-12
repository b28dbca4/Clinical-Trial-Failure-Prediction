# """
# Data Module - Clinical Trial Failure Prediction

# Complete data processing pipeline for clinical trial analysis.

# Components:
# - Data Loader: API integration and data collection
# - Data Cleaner: Cleaning, standardization, and preprocessing
# - Feature Engineer: Advanced feature creation
# - Data Validator: Quality assurance and validation
# - Data Pipeline: End-to-end orchestration

# Features:
# - Automated data collection from ClinicalTrials.gov
# - Comprehensive cleaning and preprocessing
# - Advanced feature engineering (50+ features)
# - Data quality validation
# - Memory-efficient processing
# - Progress tracking and logging
# - Caching for performance

# Meets PDF Requirements:
# - Requirement 2.1: Data collection and preprocessing
# - Professional data pipeline
# - Quality assurance
# """

# from .data_loader import ClinicalTrialsDataLoader
# from .data_cleaner import ClinicalTrialsDataCleaner
# from .feature_engineering import ClinicalTrialsFeatureEngineer
# from .data_validator import DataValidator
# from .data_pipeline import ComprehensiveDataPipeline, run_data_pipeline

# __all__ = [
#     # Main classes
#     'ClinicalTrialsDataLoader',
#     'ClinicalTrialsDataCleaner',
#     'ClinicalTrialsFeatureEngineer',
#     'DataValidator',
#     'ComprehensiveDataPipeline',
    
#     # Convenience functions
#     'run_data_pipeline',
# ]

# # Version
# __version__ = '1.0.0'

# # Quick usage guide
# USAGE_GUIDE = """
# QUICK START - DATA MODULE

# 1. Run Complete Pipeline (Easiest):
#    ---------------------------------
#    from src.data import run_data_pipeline
   
#    # Load from API and process
#    df, results = run_data_pipeline(load_from_api=True)
   
#    # Or load from file
#    df, results = run_data_pipeline(
#        load_from_api=False,
#        data_path='data/raw/clinical_trials.csv'
#    )

# 2. Step-by-Step Processing:
#    -------------------------
#    from src.data import (
#        ClinicalTrialsDataLoader,
#        ClinicalTrialsDataCleaner,
#        ClinicalTrialsFeatureEngineer
#    )
   
#    # Load data
#    loader = ClinicalTrialsDataLoader()
#    df_raw = loader.collect_comprehensive_dataset()
   
#    # Clean data
#    cleaner = ClinicalTrialsDataCleaner()
#    df_clean = cleaner.clean_dataset(df_raw)
   
#    # Engineer features
#    engineer = ClinicalTrialsFeatureEngineer()
#    df_final = engineer.engineer_features(df_clean)

# 3. With Validation:
#    ----------------
#    from src.data import ComprehensiveDataPipeline
   
#    pipeline = ComprehensiveDataPipeline()
#    df, results = pipeline.run_full_pipeline(
#        load_from_api=True,
#        save_intermediate=True
#    )
   
#    # Save processed data
#    pipeline.save_processed_data(df, 'data/processed/final.csv')
   
#    # Export validation report
#    pipeline.export_pipeline_report()

# 4. Data Validation Only:
#    ----------------------
#    from src.data import DataValidator
   
#    validator = DataValidator()
   
#    # Validate raw data
#    validation = validator.validate_raw_data(df_raw)
   
#    # Validate processed data
#    validation = validator.validate_processed_data(df_processed)
   
#    # Generate report
#    validator.generate_validation_report(validation, 'validation_report.txt')

# DETAILED USAGE:

# 1. Data Loader (ClinicalTrialsDataLoader):
#    ----------------------------------------
#    # Initialize
#    loader = ClinicalTrialsDataLoader()
   
#    # Search specific conditions
#    studies = loader.search_studies('cancer', max_records=1000)
   
#    # Extract structured data
#    df = loader.extract_detailed_study_data(studies)
   
#    # Full collection (multiple conditions)
#    df = loader.collect_comprehensive_dataset()

# 2. Data Cleaner (ClinicalTrialsDataCleaner):
#    ------------------------------------------
#    cleaner = ClinicalTrialsDataCleaner()
   
#    # Complete cleaning pipeline
#    df_clean = cleaner.clean_dataset(df_raw)
   
#    # Individual steps (if needed)
#    df = cleaner._handle_missing_values(df)
#    df = cleaner._standardize_categoricals(df)
#    df = cleaner._engineer_date_features(df)
#    df = cleaner._handle_outliers(df)
#    df = cleaner._create_target_variable(df)
#    df = cleaner._encode_categoricals(df)
   
#    # Get label encoder mappings
#    mappings = cleaner.get_label_mappings()
   
#    # Save encoders for later use
#    cleaner.save_encoders('encoders.pkl')

# 3. Feature Engineer (ClinicalTrialsFeatureEngineer):
#    -------------------------------------------------
#    engineer = ClinicalTrialsFeatureEngineer()
   
#    # Complete feature engineering
#    df_features = engineer.engineer_features(df_clean)
   
#    # Individual feature groups
#    df = engineer._create_basic_features(df)
#    df = engineer._create_interaction_features(df)
#    df = engineer._create_aggregation_features(df)
#    df = engineer._create_text_features(df)
#    df = engineer._create_time_features(df)
#    df = engineer._create_risk_features(df)
   
#    # Get feature names for modeling
#    feature_names = engineer.get_feature_names(df)

# 4. Data Validator (DataValidator):
#    --------------------------------
#    validator = DataValidator()
   
#    # Validate raw data
#    results = validator.validate_raw_data(df_raw)
#    print(f"Quality: {results['overall_quality']}")
#    print(f"Score: {results['quality_score']}/100")
   
#    # Validate processed data
#    results = validator.validate_processed_data(df_processed)
   
#    # Generate detailed report
#    report = validator.generate_validation_report(
#        results,
#        output_path='validation_report.txt'
#    )

# 5. Complete Pipeline (ComprehensiveDataPipeline):
#    ----------------------------------------------
#    pipeline = ComprehensiveDataPipeline(cache_dir='data/cache')
   
#    # Run full pipeline
#    df, results = pipeline.run_full_pipeline(
#        load_from_api=True,          # Load fresh data
#        save_intermediate=True        # Save intermediate results
#    )
   
#    # Or load existing data
#    df, results = pipeline.run_full_pipeline(
#        load_from_api=False,
#        data_path='data/raw/trials.csv'
#    )
   
#    # Prepare for modeling
#    model_ready_df = pipeline.get_feature_importance_data(df)
   
#    # Save results
#    output_path = pipeline.save_processed_data(df)
   
#    # Load processed data later
#    df_loaded = pipeline.load_processed_data(output_path)
   
#    # Export pipeline report
#    pipeline.export_pipeline_report('pipeline_report.json')

# PIPELINE FEATURES:

# Data Collection:
# - Multi-threaded API requests
# - Connection pooling and retry logic
# - Adaptive rate limiting
# - Comprehensive field extraction
# - Result caching
# - Compression for storage

# Data Cleaning:
# - Vectorized operations for speed
# - Missing value imputation (median, mode)
# - Categorical standardization
# - Date feature engineering
# - Outlier handling (IQR method)
# - Target variable creation
# - Label encoding with mapping

# Feature Engineering:
# - Basic derived features (50+)
# - Interaction features
# - Aggregation features (grouped stats)
# - Text features (TF-IDF)
# - Time-based features
# - Risk indicators
# - Polynomial features
# - Log transformations

# Data Validation:
# - Schema validation
# - Completeness checks
# - Data type validation
# - Value range checks
# - Target variable validation
# - Feature quality assessment
# - Statistical validation
# - Consistency checks
# - Quality scoring (0-100)

# Pipeline Orchestration:
# - End-to-end automation
# - Progress tracking
# - Error handling
# - Intermediate saves
# - Quality reporting
# - Memory optimization
# - Result caching

# OUTPUT FILES:

# Raw Data:
# - data/raw/clinical_trials_raw_TIMESTAMP.json.gz (compressed)

# Intermediate:
# - data/cache/clinical_trials_raw_TIMESTAMP.csv
# - data/cache/clinical_trials_cleaned_TIMESTAMP.csv
# - data/cache/clinical_trials_engineered_TIMESTAMP.csv

# Final:
# - data/processed/clinical_trials_processed_TIMESTAMP.csv
# - data/processed/clinical_trials_processed_TIMESTAMP.parquet

# Reports:
# - data/cache/pipeline_report_TIMESTAMP.json
# - validation_report.txt

# Encoders:
# - encoders.pkl (label encoder mappings)

# TYPICAL WORKFLOW:

# # 1. Collect and process data
# from src.data import run_data_pipeline
# df, results = run_data_pipeline(load_from_api=True)

# # 2. Review data quality
# print(f"Shape: {df.shape}")
# print(f"Quality: {results['final_validation']['overall_quality']}")
# print(f"Target distribution: {results['quality_report']['target_distribution']}")

# # 3. Prepare for modeling
# from sklearn.model_selection import train_test_split
# X = df.drop(['is_successful'], axis=1)
# y = df['is_successful']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # 4. Train models (see models module)
# from src.models import train_and_compare_all_models
# model_results, best_model = train_and_compare_all_models(
#     X_train, y_train, X_test, y_test
# )

# PERFORMANCE TIPS:

# 1. Use caching for repeated runs:
#    - Pipeline automatically caches API results
#    - Reuse processed data instead of reprocessing

# 2. Memory optimization:
#    - Use parquet format for large datasets
#    - Process in chunks if memory constrained

# 3. Speed optimization:
#    - Parallel processing enabled by default
#    - Use load_from_api=False for existing data

# 4. Quality assurance:
#    - Always check validation results
#    - Review quality score before modeling
#    - Inspect high missing value columns

# TROUBLESHOOTING:

# Issue: API rate limiting
# Solution: Increase delays in loader, reduce max_workers

# Issue: Memory errors
# Solution: Process fewer records, use sampling

# Issue: Missing required columns
# Solution: Check API response format, update field extraction

# Issue: Low quality score
# Solution: Review validation report, improve data cleaning

# Issue: Target imbalance
# Solution: Use balancing techniques in model training (SMOTE, etc.)
# """

# def print_usage():
#     """Print usage guide"""
#     print(USAGE_GUIDE)


# def get_pipeline_components():
#     """Get list of all pipeline components"""
#     return {
#         'loaders': ['ClinicalTrialsDataLoader'],
#         'cleaners': ['ClinicalTrialsDataCleaner'],
#         'engineers': ['ClinicalTrialsFeatureEngineer'],
#         'validators': ['DataValidator'],
#         'pipelines': ['ComprehensiveDataPipeline'],
#         'utilities': ['run_data_pipeline']
#     }


# def get_feature_categories():
#     """Get categories of engineered features"""
#     return {
#         'basic_features': [
#             'enrollment_per_day',
#             'enrollment_per_month',
#             'enrollment_per_facility',
#             'facilities_per_country',
#             'log transformations',
#             'polynomial features'
#         ],
#         'interaction_features': [
#             'phase_enrollment_interaction',
#             'studytype_duration_interaction',
#             'sponsor_facility_interaction'
#         ],
#         'aggregation_features': [
#             'phase_enrollment_stats',
#             'studytype_duration_stats',
#             'sponsor_stats'
#         ],
#         'text_features': [
#             'title_length',
#             'title_word_count',
#             'conditions_count',
#             'intervention_types_count',
#             'condition_tfidf_features'
#         ],
#         'time_features': [
#             'start_year',
#             'start_month',
#             'start_quarter',
#             'start_day_of_week',
#             'start_is_weekend',
#             'years_since_start',
#             'start_decade',
#             'is_recent_study',
#             'start_season',
#             'duration_category'
#         ],
#         'risk_features': [
#             'complexity_score',
#             'is_early_phase',
#             'is_small_enrollment',
#             'is_multi_country'
#         ]
#     }
