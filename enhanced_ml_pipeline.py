#=============================================================================
# enhanced_ml_pipeline.py
"""
Enhanced ML Pipeline with comprehensive configuration and error handling
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)

@dataclass
class MLConfig:
    """Configuration class for ML pipeline settings"""
    enable_hyperparameter_tuning: bool = True
    enable_shap: bool = True
    enable_feature_selection: bool = True
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    
    # Model-specific configurations
    rf_config: Dict[str, Any] = None
    gb_config: Dict[str, Any] = None
    ridge_config: Dict[str, Any] = None
    svr_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set default model configurations"""
        if self.rf_config is None:
            self.rf_config = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
            }
        
        if self.gb_config is None:
            self.gb_config = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': self.random_state
            }
        
        if self.ridge_config is None:
            self.ridge_config = {
                'alpha': 1.0,
                'random_state': self.random_state
            }
        
        if self.svr_config is None:
            self.svr_config = {
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            }

class EnhancedMLPipeline:
    """Enhanced ML Pipeline with comprehensive analysis capabilities"""
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.models = {}
        self.results = {}
        self.feature_names = []
        
    def prepare_data(self, data: pd.DataFrame, target_col: str, 
                    feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ML analysis"""
        try:
            if feature_cols is None:
                # Auto-detect numeric features
                feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if target_col in feature_cols:
                    feature_cols.remove(target_col)
            
            X = data[feature_cols].copy()
            y = data[target_col].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Store feature names
            self.feature_names = feature_cols
            
            logger.info(f"Prepared data: {len(X)} samples, {len(feature_cols)} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def validate_data(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Validate input data"""
        try:
            # Check for empty data
            if X.empty or y.empty:
                logger.error("Empty input data")
                return False
            
            # Check dimensions match
            if len(X) != len(y):
                logger.error("Feature and target dimensions don't match")
                return False
            
            # Check for sufficient samples
            if len(X) < 10:
                logger.warning("Very small dataset - results may be unreliable")
            
            # Check for constant features
            constant_features = X.columns[X.var() == 0].tolist()
            if constant_features:
                logger.warning(f"Constant features detected: {constant_features}")
            
            # Check for missing values
            if X.isnull().any().any() or y.isnull().any():
                logger.warning("Missing values detected - will be handled during processing")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def get_model_configs(self) -> Dict[str, Dict]:
        """Get model configurations"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.svm import SVR
        
        return {
            'Random Forest': {
                'model_class': RandomForestRegressor,
                'params': self.config.rf_config,
                'hyperopt_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                } if self.config.enable_hyperparameter_tuning else {}
            },
            'Gradient Boosting': {
                'model_class': GradientBoostingRegressor,
                'params': self.config.gb_config,
                'hyperopt_grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                } if self.config.enable_hyperparameter_tuning else {}
            },
            'Ridge Regression': {
                'model_class': Ridge,
                'params': self.config.ridge_config,
                'hyperopt_grid': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                } if self.config.enable_hyperparameter_tuning else {}
            },
            'Support Vector Regression': {
                'model_class': SVR,
                'params': self.config.svr_config,
                'hyperopt_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.01, 0.1, 0.2]
                } if self.config.enable_hyperparameter_tuning else {}
            }
        }
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    models_to_train: List[str] = None) -> Dict[str, Any]:
        """Train multiple ML models"""
        try:
            from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            if not self.validate_data(X, y):
                raise ValueError("Data validation failed")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, 
                random_state=self.config.random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Get model configurations
            model_configs = self.get_model_configs()
            
            # Filter models to train
            if models_to_train is None:
                models_to_train = list(model_configs.keys())
            
            results = {
                'models': {},
                'metadata': {
                    'feature_names': self.feature_names,
                    'scaler': scaler,
                    'X_train': X_train_scaled,
                    'X_test': X_test_scaled,
                    'y_train': y_train,
                    'y_test': y_test,
                    'config': self.config
                }
            }
            
            # Train each model
            for model_name in models_to_train:
                if model_name not in model_configs:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                try:
                    config = model_configs[model_name]
                    
                    # Initialize model
                    model = config['model_class'](**config['params'])
                    
                    # Hyperparameter tuning
                    if self.config.enable_hyperparameter_tuning and config['hyperopt_grid']:
                        grid_search = GridSearchCV(
                            model, config['hyperopt_grid'],
                            cv=self.config.cv_folds,
                            scoring='r2',
                            n_jobs=self.config.n_jobs,
                            verbose=0
                        )
                        grid_search.fit(X_train_scaled, y_train)
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        cv_results = grid_search.cv_results_
                    else:
                        best_model = model
                        best_model.fit(X_train_scaled, y_train)
                        best_params = config['params']
                        cv_results = None
                    
                    # Generate predictions
                    y_pred_train = best_model.predict(X_train_scaled)
                    y_pred_test = best_model.predict(X_test_scaled)
                    
                    # Cross-validation scores
                    cv_scores = cross_val_score(
                        best_model, X_train_scaled, y_train,
                        cv=self.config.cv_folds, scoring='r2'
                    )
                    
                    # Calculate metrics
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    
                    # Store model results
                    model_result = {
                        'model': best_model,
                        'model_name': model_name,
                        'best_params': best_params,
                        'cv_results': cv_results,
                        'predictions': {
                            'train': y_pred_train,
                            'test': y_pred_test
                        },
                        'scores': {
                            'train_r2': train_r2,
                            'test_r2': test_r2,
                            'train_rmse': train_rmse,
                            'test_rmse': test_rmse,
                            'train_mae': train_mae,
                            'test_mae': test_mae,
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std(),
                            'cv_scores': cv_scores
                        }
                    }
                    
                    # Add feature importance if available
                    if hasattr(best_model, 'feature_importances_'):
                        model_result['feature_importances'] = best_model.feature_importances_
                    elif hasattr(best_model, 'coef_'):
                        model_result['feature_importances'] = np.abs(best_model.coef_)
                    
                    results['models'][model_name] = model_result
                    logger.info(f"Successfully trained {model_name}: Test R² = {test_r2:.3f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {str(e)}")
                    continue
            
            self.results = results
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def generate_explanations(self, model_name: str, method: str = 'shap') -> Optional[Any]:
        """Generate model explanations using SHAP or LIME"""
        try:
            if model_name not in self.results['models']:
                logger.error(f"Model {model_name} not found in results")
                return None
            
            model_result = self.results['models'][model_name]
            model = model_result['model']
            metadata = self.results['metadata']
            
            if method.lower() == 'shap':
                return self._generate_shap_explanations(model, metadata)
            elif method.lower() == 'lime':
                return self._generate_lime_explanations(model, metadata)
            else:
                logger.error(f"Unknown explanation method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            return None
    
    def _generate_shap_explanations(self, model, metadata: Dict) -> Optional[np.ndarray]:
        """Generate SHAP explanations"""
        try:
            import shap
            
            X_test = metadata['X_test']
            X_train = metadata['X_train']
            
            # Choose appropriate explainer
            if hasattr(model, 'tree_'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test.iloc[:min(100, len(X_test))])
            elif hasattr(model, 'coef_'):
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer.shap_values(X_test.iloc[:min(100, len(X_test))])
            else:
                # Use KernelExplainer for other models
                sample_size = min(50, len(X_train))
                explainer = shap.KernelExplainer(model.predict, X_train.sample(sample_size))
                shap_values = explainer.shap_values(X_test.iloc[:min(20, len(X_test))])
            
            logger.info("SHAP explanations generated successfully")
            return shap_values
            
        except ImportError:
            logger.warning("SHAP not available - install with: pip install shap")
            return None
        except Exception as e:
            logger.error(f"SHAP explanation failed: {str(e)}")
            return None
    
    def _generate_lime_explanations(self, model, metadata: Dict) -> Optional[List[Dict]]:
        """Generate LIME explanations"""
        try:
            import lime
            import lime.lime_tabular
            
            X_train = metadata['X_train']
            X_test = metadata['X_test']
            feature_names = metadata['feature_names']
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                class_names=['target'],
                mode='regression',
                discretize_continuous=True
            )
            
            # Generate explanations for sample instances
            explanations = []
            n_instances = min(5, len(X_test))
            
            for i in range(n_instances):
                exp = explainer.explain_instance(
                    X_test.iloc[i].values,
                    model.predict,
                    num_features=len(feature_names)
                )
                
                explanation_data = {
                    'instance_idx': i,
                    'prediction': model.predict(X_test.iloc[i:i+1])[0],
                    'explanation': exp.as_list(),
                    'score': exp.score
                }
                
                explanations.append(explanation_data)
            
            logger.info("LIME explanations generated successfully")
            return explanations
            
        except ImportError:
            logger.warning("LIME not available - install with: pip install lime")
            return None
        except Exception as e:
            logger.error(f"LIME explanation failed: {str(e)}")
            return None
    
    def get_best_model(self, metric: str = 'test_r2') -> Tuple[str, Dict]:
        """Get the best performing model based on specified metric"""
        if not self.results or 'models' not in self.results:
            raise ValueError("No trained models available")
        
        best_score = float('-inf')
        best_model_name = None
        best_model_result = None
        
        for model_name, result in self.results['models'].items():
            if metric in result['scores']:
                score = result['scores'][metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model_result = result
        
        if best_model_name is None:
            raise ValueError(f"Metric {metric} not found in results")
        
        return best_model_name, best_model_result
    
    def generate_model_report(self) -> str:
        """Generate a comprehensive model performance report"""
        if not self.results or 'models' not in self.results:
            return "No trained models available for reporting"
        
        report = "# Machine Learning Model Performance Report\n\n"
        
        # Summary table
        report += "## Model Performance Summary\n\n"
        report += "| Model | Test R² | Test RMSE | Test MAE | CV Mean | CV Std |\n"
        report += "|-------|---------|-----------|----------|---------|--------|\n"
        
        for model_name, result in self.results['models'].items():
            scores = result['scores']
            report += f"| {model_name} | {scores['test_r2']:.4f} | {scores['test_rmse']:.4f} | {scores['test_mae']:.4f} | {scores['cv_mean']:.4f} | {scores['cv_std']:.4f} |\n"
        
        # Best model details
        try:
            best_model_name, best_result = self.get_best_model()
            report += f"\n## Best Model: {best_model_name}\n\n"
            report += f"**Test R²:** {best_result['scores']['test_r2']:.4f}\n"
            report += f"**Test RMSE:** {best_result['scores']['test_rmse']:.4f}\n"
            report += f"**Cross-Validation:** {best_result['scores']['cv_mean']:.4f} ± {best_result['scores']['cv_std']:.4f}\n"
            
            # Hyperparameters
            if best_result['best_params']:
                report += f"\n**Best Hyperparameters:**\n"
                for param, value in best_result['best_params'].items():
                    report += f"- {param}: {value}\n"
            
        except Exception as e:
            report += f"\nError generating best model details: {str(e)}\n"
        
        # Configuration details
        report += f"\n## Configuration\n\n"
        report += f"- **Test Set Size:** {self.config.test_size}\n"
        report += f"- **Cross-Validation Folds:** {self.config.cv_folds}\n"
        report += f"- **Random State:** {self.config.random_state}\n"
        report += f"- **Hyperparameter Tuning:** {'Enabled' if self.config.enable_hyperparameter_tuning else 'Disabled'}\n"
        
        return report
    
    def save_results(self, filepath: str, format: str = 'pickle'):
        """Save model results to file"""
        try:
            if format.lower() == 'pickle':
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(self.results, f)
            elif format.lower() == 'json':
                import json
                # Convert numpy arrays to lists for JSON serialization
                json_results = self._prepare_for_json(self.results)
                with open(filepath, 'w') as f:
                    json.dump(json_results, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items() if k != 'model'}
        elif isinstance(obj, (list, tuple)):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            return obj
    
    def load_results(self, filepath: str, format: str = 'pickle'):
        """Load model results from file"""
        try:
            if format.lower() == 'pickle':
                import pickle
                with open(filepath, 'rb') as f:
                    self.results = pickle.load(f)
            elif format.lower() == 'json':
                import json
                with open(filepath, 'r') as f:
                    self.results = json.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Results loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load results: {str(e)}")
            raise
    
    def clear_results(self):
        """Clear stored results and models"""
        self.results = {}
        self.models = {}
        self.feature_names = []
        logger.info("Results cleared")

# Utility functions for the pipeline
def create_default_config(**kwargs) -> MLConfig:
    """Create a default ML configuration with optional overrides"""
    return MLConfig(**kwargs)

def compare_model_performance(results1: Dict, results2: Dict, metric: str = 'test_r2') -> pd.DataFrame:
    """Compare performance between two sets of model results"""
    comparison_data = []
    
    for model_name in set(results1.get('models', {}).keys()) | set(results2.get('models', {}).keys()):
        row = {'Model': model_name}
        
        if model_name in results1.get('models', {}):
            row['Results1_' + metric] = results1['models'][model_name]['scores'].get(metric, np.nan)
        else:
            row['Results1_' + metric] = np.nan
            
        if model_name in results2.get('models', {}):
            row['Results2_' + metric] = results2['models'][model_name]['scores'].get(metric, np.nan)
        else:
            row['Results2_' + metric] = np.nan
        
        # Calculate improvement
        if not (np.isnan(row['Results1_' + metric]) or np.isnan(row['Results2_' + metric])):
            row['Improvement'] = row['Results2_' + metric] - row['Results1_' + metric]
        else:
            row['Improvement'] = np.nan
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Generate synthetic data
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some relationship to features
    y = pd.Series(
        X['feature_0'] * 2 + X['feature_1'] * 1.5 - X['feature_2'] * 0.5 + np.random.randn(n_samples) * 0.1,
        name='target'
    )
    
    # Test the pipeline
    config = MLConfig(enable_hyperparameter_tuning=False, cv_folds=3)
    pipeline = EnhancedMLPipeline(config)
    
    print("Testing Enhanced ML Pipeline...")
    
    try:
        # Train models
        results = pipeline.train_models(X, y, models_to_train=['Random Forest', 'Ridge Regression'])
        
        # Get best model
        best_model_name, best_result = pipeline.get_best_model()
        print(f"Best model: {best_model_name} (R² = {best_result['scores']['test_r2']:.3f})")
        
        # Generate report
        report = pipeline.generate_model_report()
        print("\nGenerated model report successfully")
        
        print("Pipeline test completed successfully!")
        
    except Exception as e:
        print(f"Pipeline test failed: {str(e)}")
        logger.error(f"Pipeline test error: {str(e)}")