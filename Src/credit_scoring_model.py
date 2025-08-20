"""
Enhanced Credit Scoring Model with professional architecture and error handling.
Uses shared utilities to eliminate code duplication and improve maintainability.
"""

import sys
import os
import argparse
from datetime import datetime

# Add src to path for importing utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    load_and_validate_dataset,
    preprocess_credit_data, 
    prepare_features_and_target,
    split_and_scale_data,
    evaluate_model_performance,
    save_model_and_scaler
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_credit_scoring_model(data_path: str = 'Data/Credit Score Classification Dataset.csv',
                             model_dir: str = 'models',
                             test_size: float = 0.2,
                             random_state: int = 42,
                             perform_tuning: bool = True) -> dict:
    """
    Complete credit scoring model training pipeline.
    
    Args:
        data_path: Path to the dataset
        model_dir: Directory to save trained models
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        perform_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        Dictionary with training results and metrics
    """
    try:
        logger.info("Starting Credit Scoring Model Training Pipeline")
        start_time = datetime.now()
        
        # Step 1: Load and validate dataset
        logger.info("Step 1: Loading and validating dataset...")
        data = load_and_validate_dataset(data_path)
        
        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing data...")
        processed_data, preprocessing_info = preprocess_credit_data(data)
        
        # Step 3: Prepare features and target
        logger.info("Step 3: Preparing features and target...")
        X, y = prepare_features_and_target(processed_data)
        
        # Step 4: Split and scale data
        logger.info("Step 4: Splitting and scaling data...")
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Step 5: Train base model
        logger.info("Step 5: Training Random Forest model...")
        rf_model = RandomForestClassifier(random_state=random_state)
        rf_model.fit(X_train, y_train)
        
        # Step 6: Hyperparameter tuning (optional)
        if perform_tuning:
            logger.info("Step 6: Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            best_model = rf_model
            logger.info("Skipping hyperparameter tuning, using default model")
        
        # Step 7: Evaluate model
        logger.info("Step 7: Evaluating model performance...")
        evaluation_results = evaluate_model_performance(best_model, X_test, y_test)
        
        # Step 8: Save model and scaler
        logger.info("Step 8: Saving model and scaler...")
        saved_paths = save_model_and_scaler(best_model, scaler, model_dir)
        
        # Compile results
        training_results = {
            'model': best_model,
            'scaler': scaler,
            'preprocessing_info': preprocessing_info,
            'evaluation_results': evaluation_results,
            'saved_paths': saved_paths,
            'training_time': str(datetime.now() - start_time),
            'feature_names': list(X.columns),
            'target_mapping': {'Low': 0, 'Average': 1, 'High': 2}
        }
        
        logger.info("Credit Scoring Model Training Pipeline Completed Successfully!")
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"ROC-AUC Score: {evaluation_results['roc_auc_score']:.4f}")
        print(f"Model saved to: {saved_paths['model_path']}")
        print(f"Scaler saved to: {saved_paths['scaler_path']}")
        print("="*80)
        
        return training_results
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description='Credit Scoring Model Training')
    parser.add_argument('--data-path', default='Data/Credit Score Classification Dataset.csv',
                      help='Path to the dataset CSV file')
    parser.add_argument('--model-dir', default='models',
                      help='Directory to save trained models')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of data for testing (0.0-1.0)')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--no-tuning', action='store_true',
                      help='Skip hyperparameter tuning for faster training')
    
    args = parser.parse_args()
    
    try:
        # Run training pipeline
        results = train_credit_scoring_model(
            data_path=args.data_path,
            model_dir=args.model_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            perform_tuning=not args.no_tuning
        )
        
        print("\nTraining completed successfully!")
        print(f"Check {args.model_dir}/ for saved models")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()