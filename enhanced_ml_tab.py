#=============================================================================
# enhanced_ml_tab.py
"""
Enhanced ML Analysis Tab with comprehensive machine learning capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import logging
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
import time

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedMLAnalysisTab:
    """Enhanced ML Analysis Tab with both SHAP and LIME explanations"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.color_palette = {
            'primary': '#00d4aa',
            'secondary': '#ffab00', 
            'danger': '#ff5252',
            'info': '#2196f3',
            'success': '#4caf50',
            'warning': '#ff9500'
        }
    
    def render(self, data: pd.DataFrame, hit_threshold: float):
        """
        Render the Enhanced ML Analysis tab
        
        Args:
            data: HTS screening data
            hit_threshold: Threshold for hit identification
        """
        st.markdown("### Advanced Machine Learning Analysis")
        
        if data is None or len(data) == 0:
            st.warning("No data available for ML analysis.")
            return
        
        # Check for required features
        feature_cols = [col for col in data.columns if col in [
            'MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'AromaticRings', 
            'Complexity', 'SlogP', 'NumHeavyAtoms', 'FractionCsp3', 'NumAliphaticRings'
        ]]
        
        if len(feature_cols) == 0:
            st.error("No chemical descriptors found for ML analysis. Required features: MW, LogP, HBD, HBA, etc.")
            return
        
        try:
            # Prepare data
            X, y, feature_names = self._prepare_ml_data(data, feature_cols)
            
            if X is None or len(X) == 0:
                st.error("Insufficient data for ML analysis")
                return
            
            # UI Controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                models_to_run = st.multiselect(
                    "Select Models",
                    ['Random Forest', 'Gradient Boosting', 'Ridge Regression', 'Support Vector Regression'],
                    default=['Random Forest', 'Gradient Boosting']
                )
            
            with col2:
                test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
                enable_hyperopt = st.checkbox("Hyperparameter Tuning", value=True)
            
            with col3:
                explanation_method = st.selectbox(
                    "Explanation Method",
                    ["None", "SHAP", "LIME", "Both SHAP & LIME"],
                    index=3
                )
                cross_val_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            
            # Run analysis button
            if st.button("Run ML Analysis", type="primary"):
                with st.spinner("Training models and generating explanations..."):
                    results = self._run_ml_analysis(
                        X, y, feature_names, models_to_run, test_size, 
                        enable_hyperopt, explanation_method, cross_val_folds
                    )
                    
                    if results:
                        self._display_results(results, explanation_method)
                    else:
                        st.error("ML analysis failed. Check the logs for details.")
        
        except Exception as e:
            st.error(f"ML Analysis Error: {str(e)}")
            logger.error(f"Enhanced ML tab error: {str(e)}\n{traceback.format_exc()}")
    
    def _prepare_ml_data(self, data: pd.DataFrame, feature_cols: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
        """Prepare data for ML analysis"""
        try:
            # Filter for test compounds only (exclude controls)
            test_compounds = data[~data['Compound'].str.contains('Control|DMSO', case=False, na=False)]
            
            if len(test_compounds) < 10:
                st.warning("Insufficient test compounds for reliable ML analysis (need ≥10)")
                return None, None, []
            
            # Prepare features and target
            X = test_compounds[feature_cols].copy()
            y = test_compounds['Expression_Fold_Change'].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Remove constant features
            constant_features = X.columns[X.var() == 0].tolist()
            if constant_features:
                X = X.drop(columns=constant_features)
                st.info(f"Removed constant features: {constant_features}")
            
            # Remove highly correlated features (>0.95)
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            
            if to_drop:
                X = X.drop(columns=to_drop)
                st.info(f"Removed highly correlated features: {to_drop}")
            
            logger.info(f"Prepared ML data: {len(X)} samples, {len(X.columns)} features")
            return X, y, X.columns.tolist()
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            return None, None, []
    
    def _run_ml_analysis(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str],
                        models_to_run: List[str], test_size: float, enable_hyperopt: bool,
                        explanation_method: str, cv_folds: int) -> Dict[str, Any]:
        """Run comprehensive ML analysis"""
        try:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=None
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
            
            results = {
                'models': {},
                'metadata': {
                    'feature_names': feature_names,
                    'X_train': X_train_scaled,
                    'X_test': X_test_scaled,
                    'y_train': y_train,
                    'y_test': y_test,
                    'scaler': scaler
                }
            }
            
            # Define models
            model_configs = {
                'Random Forest': {
                    'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5]
                    } if enable_hyperopt else {}
                },
                'Gradient Boosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    } if enable_hyperopt else {}
                },
                'Ridge Regression': {
                    'model': Ridge(random_state=42),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0, 100.0]
                    } if enable_hyperopt else {}
                },
                'Support Vector Regression': {
                    'model': SVR(),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'gamma': ['scale', 'auto', 0.01],
                        'epsilon': [0.01, 0.1, 0.2]
                    } if enable_hyperopt else {}
                }
            }
            
            # Train models
            for model_name in models_to_run:
                if model_name not in model_configs:
                    continue
                
                try:
                    config = model_configs[model_name]
                    model = config['model']
                    
                    # Hyperparameter tuning
                    if enable_hyperopt and config['params']:
                        grid_search = GridSearchCV(
                            model, config['params'], 
                            cv=cv_folds, scoring='r2', 
                            n_jobs=-1, verbose=0
                        )
                        grid_search.fit(X_train_scaled, y_train)
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                    else:
                        best_model = model
                        best_model.fit(X_train_scaled, y_train)
                        best_params = {}
                    
                    # Predictions
                    y_pred_train = best_model.predict(X_train_scaled)
                    y_pred_test = best_model.predict(X_test_scaled)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        best_model, X_train_scaled, y_train, 
                        cv=cv_folds, scoring='r2'
                    )
                    
                    # Calculate metrics
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    
                    # Store results
                    model_result = {
                        'model': best_model,
                        'best_params': best_params,
                        'predictions_train': y_pred_train,
                        'predictions_test': y_pred_test,
                        'scores': {
                            'train_r2': train_r2,
                            'test_r2': test_r2,
                            'train_rmse': train_rmse,
                            'test_rmse': test_rmse,
                            'train_mae': train_mae,
                            'test_mae': test_mae,
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std()
                        },
                        'cv_scores': cv_scores
                    }
                    
                    # Generate explanations
                    if explanation_method in ['SHAP', 'Both SHAP & LIME']:
                        model_result['shap_values'] = self._generate_shap_explanations(
                            best_model, X_train_scaled, X_test_scaled, model_name
                        )
                    
                    if explanation_method in ['LIME', 'Both SHAP & LIME']:
                        model_result['lime_explanations'] = self._generate_lime_explanations(
                            best_model, X_train_scaled, X_test_scaled, feature_names, model_name
                        )
                    
                    results['models'][model_name] = model_result
                    logger.info(f"Successfully trained {model_name}: Test R² = {test_r2:.3f}")
                    
                except Exception as e:
                    logger.error(f"Model {model_name} training failed: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"ML analysis failed: {str(e)}")
            return {}
    
    def _generate_shap_explanations(self, model, X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame, model_name: str) -> Optional[np.ndarray]:
        """Generate SHAP explanations"""
        try:
            import shap
            
            # Choose explainer based on model type
            if hasattr(model, 'tree_'):
                # Tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test.iloc[:min(100, len(X_test))])
            elif hasattr(model, 'coef_'):
                # Linear models
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer.shap_values(X_test.iloc[:min(100, len(X_test))])
            else:
                # Other models - use KernelExplainer (slower)
                explainer = shap.KernelExplainer(model.predict, X_train.sample(min(50, len(X_train))))
                shap_values = explainer.shap_values(X_test.iloc[:min(20, len(X_test))])
            
            logger.info(f"Generated SHAP explanations for {model_name}")
            return shap_values
            
        except ImportError:
            st.warning("SHAP not installed. Run: pip install shap")
            return None
        except Exception as e:
            logger.warning(f"SHAP explanation failed for {model_name}: {str(e)}")
            return None
    
    def _generate_lime_explanations(self, model, X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame, feature_names: List[str], 
                                   model_name: str) -> Optional[List[Dict]]:
        """Generate LIME explanations"""
        try:
            import lime
            import lime.lime_tabular
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                class_names=['Expression_Fold_Change'],
                mode='regression',
                discretize_continuous=True
            )
            
            # Generate explanations for a few instances
            explanations = []
            n_instances = min(5, len(X_test))
            
            for i in range(n_instances):
                exp = explainer.explain_instance(
                    X_test.iloc[i].values,
                    model.predict,
                    num_features=len(feature_names)
                )
                
                # Extract explanation data
                explanation_data = {
                    'instance_idx': i,
                    'prediction': model.predict(X_test.iloc[i:i+1])[0],
                    'features': [],
                    'importance': []
                }
                
                for feature_idx, importance in exp.as_list():
                    if isinstance(feature_idx, str):
                        # Handle categorical features
                        feature_name = feature_idx.split(' ')[0]
                    else:
                        # Handle numerical features
                        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature_{feature_idx}'
                    
                    explanation_data['features'].append(feature_name)
                    explanation_data['importance'].append(importance)
                
                explanations.append(explanation_data)
            
            logger.info(f"Generated LIME explanations for {model_name}")
            return explanations
            
        except ImportError:
            st.warning("LIME not installed. Run: pip install lime")
            return None
        except Exception as e:
            logger.warning(f"LIME explanation failed for {model_name}: {str(e)}")
            return None
    
    def _display_results(self, results: Dict[str, Any], explanation_method: str):
        """Display ML analysis results"""
        try:
            models = results['models']
            if not models:
                st.error("No models were successfully trained")
                return
            
            # Model comparison
            st.markdown("### Model Performance Comparison")
            
            # Create comparison DataFrame
            comparison_data = []
            for model_name, result in models.items():
                scores = result['scores']
                comparison_data.append({
                    'Model': model_name,
                    'Test R²': scores['test_r2'],
                    'Test RMSE': scores['test_rmse'], 
                    'Test MAE': scores['test_mae'],
                    'CV Mean': scores['cv_mean'],
                    'CV Std': scores['cv_std']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display metrics table
            st.dataframe(
                comparison_df.round(4), 
                use_container_width=True,
                hide_index=True
            )
            
            # Best model identification
            best_model_name = comparison_df.loc[comparison_df['Test R²'].idxmax(), 'Model']
            st.success(f"Best Model: **{best_model_name}** (R² = {comparison_df['Test R²'].max():.3f})")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # R² comparison
                fig_r2 = px.bar(
                    comparison_df, x='Model', y='Test R²',
                    title='Test R² Comparison',
                    color='Test R²',
                    color_continuous_scale='Viridis'
                )
                fig_r2.update_layout(template='plotly_dark', font=dict(color='white'))
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with col2:
                # Cross-validation performance
                fig_cv = go.Figure()
                fig_cv.add_trace(go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df['CV Mean'],
                    error_y=dict(type='data', array=comparison_df['CV Std']),
                    marker_color=self.color_palette['primary'],
                    name='CV Performance'
                ))
                fig_cv.update_layout(
                    title='Cross-Validation Performance',
                    xaxis_title='Model',
                    yaxis_title='CV R²',
                    template='plotly_dark',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_cv, use_container_width=True)
            
            # Detailed analysis for best model
            st.markdown(f"### Detailed Analysis - {best_model_name}")
            
            best_result = models[best_model_name]
            metadata = results['metadata']
            
            # Prediction vs actual plot
            y_test = metadata['y_test']
            y_pred = best_result['predictions_test']
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=y_test, y=y_pred,
                mode='markers',
                marker=dict(color=self.color_palette['primary'], size=8, opacity=0.7),
                name='Predictions'
            ))
            
            # Perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                line=dict(color=self.color_palette['danger'], dash='dash'),
                name='Perfect Prediction'
            ))
            
            fig_pred.update_layout(
                title=f'Predictions vs Actual - {best_model_name}',
                xaxis_title='Actual Expression Fold Change',
                yaxis_title='Predicted Expression Fold Change',
                template='plotly_dark',
                font=dict(color='white')
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Feature importance (for tree-based models)
            if hasattr(best_result['model'], 'feature_importances_'):
                self._display_feature_importance(best_result['model'], metadata['feature_names'])
            
            # Model explanations
            if explanation_method != "None":
                st.markdown("### Model Explanations")
                
                explanation_tabs = []
                if 'shap_values' in best_result and best_result['shap_values'] is not None:
                    explanation_tabs.append("SHAP")
                if 'lime_explanations' in best_result and best_result['lime_explanations'] is not None:
                    explanation_tabs.append("LIME")
                
                if explanation_tabs:
                    if len(explanation_tabs) == 1:
                        # Single explanation method
                        if explanation_tabs[0] == "SHAP":
                            self._display_shap_explanations(best_result, metadata)
                        else:
                            self._display_lime_explanations(best_result)
                    else:
                        # Multiple explanation methods - use tabs
                        shap_tab, lime_tab = st.tabs(["SHAP Analysis", "LIME Analysis"])
                        
                        with shap_tab:
                            self._display_shap_explanations(best_result, metadata)
                        
                        with lime_tab:
                            self._display_lime_explanations(best_result)
            
            # Hyperparameter information
            if best_result['best_params']:
                st.markdown("### Best Hyperparameters")
                st.json(best_result['best_params'])
            
        except Exception as e:
            st.error(f"Results display failed: {str(e)}")
            logger.error(f"Results display error: {str(e)}\n{traceback.format_exc()}")
    
    def _display_feature_importance(self, model, feature_names: List[str]):
        """Display feature importance for tree-based models"""
        try:
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                feature_importance_df.head(10),
                x='Importance', y='Feature',
                orientation='h',
                title='Top 10 Feature Importances',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(template='plotly_dark', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Feature importance display failed: {str(e)}")
    
    def _display_shap_explanations(self, result: Dict[str, Any], metadata: Dict[str, Any]):
        """Display SHAP explanations"""
        try:
            shap_values = result['shap_values']
            feature_names = metadata['feature_names']
            
            if shap_values is None:
                st.warning("No SHAP values available")
                return
            
            # Handle multi-dimensional SHAP values
            if len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 0]
            
            st.markdown("#### SHAP Feature Importance")
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean_Abs_SHAP': mean_abs_shap
            }).sort_values('Mean_Abs_SHAP', ascending=False)
            
            # SHAP importance plot
            fig_shap = px.bar(
                shap_importance_df.head(10),
                x='Mean_Abs_SHAP', y='Feature',
                orientation='h',
                title='SHAP Feature Importance (Mean |SHAP value|)',
                color='Mean_Abs_SHAP',
                color_continuous_scale='Plasma'
            )
            fig_shap.update_layout(template='plotly_dark', font=dict(color='white'))
            st.plotly_chart(fig_shap, use_container_width=True)
            
            # SHAP waterfall for first instance
            if len(shap_values) > 0:
                st.markdown("#### SHAP Waterfall Plot (First Instance)")
                instance_shap = shap_values[0]
                X_test = metadata['X_test']
                instance_values = X_test.iloc[0]
                
                # Sort by absolute SHAP value
                sorted_indices = np.argsort(np.abs(instance_shap))[::-1][:10]
                
                waterfall_data = []
                for idx in sorted_indices:
                    feature_name = feature_names[idx]
                    shap_val = instance_shap[idx]
                    feature_val = instance_values.iloc[idx]
                    
                    waterfall_data.append({
                        'Feature': f"{feature_name}={feature_val:.2f}",
                        'SHAP_Value': shap_val,
                        'Color': 'Positive' if shap_val > 0 else 'Negative'
                    })
                
                waterfall_df = pd.DataFrame(waterfall_data)
                
                fig_waterfall = px.bar(
                    waterfall_df, x='Feature', y='SHAP_Value',
                    color='Color',
                    color_discrete_map={'Positive': self.color_palette['success'], 'Negative': self.color_palette['danger']},
                    title='SHAP Values for Individual Prediction'
                )
                fig_waterfall.update_layout(
                    template='plotly_dark',
                    font=dict(color='white'),
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_waterfall, use_container_width=True)
            
        except Exception as e:
            st.error(f"SHAP display failed: {str(e)}")
            logger.error(f"SHAP display error: {str(e)}")
    
    def _display_lime_explanations(self, result: Dict[str, Any]):
        """Display LIME explanations"""
        try:
            lime_explanations = result['lime_explanations']
            
            if lime_explanations is None:
                st.warning("No LIME explanations available")
                return
            
            st.markdown("#### LIME Local Explanations")
            
            # Display explanations for each instance
            for i, explanation in enumerate(lime_explanations):
                with st.expander(f"Instance {explanation['instance_idx']} (Prediction: {explanation['prediction']:.3f})"):
                    
                    # Create explanation DataFrame
                    lime_df = pd.DataFrame({
                        'Feature': explanation['features'],
                        'Importance': explanation['importance']
                    }).sort_values('Importance', key=abs, ascending=False)
                    
                    # LIME explanation plot
                    fig_lime = px.bar(
                        lime_df.head(10), x='Importance', y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='RdYlBu_r',
                        title=f'LIME Explanation - Instance {explanation["instance_idx"]}'
                    )
                    fig_lime.update_layout(template='plotly_dark', font=dict(color='white'))
                    st.plotly_chart(fig_lime, use_container_width=True)
            
            # Comparison summary
            st.markdown("#### LIME vs SHAP Comparison")
            st.info("""
            **LIME vs SHAP:**
            - **LIME**: Local explanations, faster, model-agnostic, explains individual predictions
            - **SHAP**: Global + local explanations, theoretically grounded, unified framework
            - **Use Case**: LIME for quick local insights, SHAP for comprehensive analysis
            """)
            
        except Exception as e:
            st.error(f"LIME display failed: {str(e)}")
            logger.error(f"LIME display error: {str(e)}")

# Additional utility functions for ML analysis
def calculate_feature_correlations(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Calculate feature-target correlations"""
    correlations = []
    for col in X.columns:
        corr = X[col].corr(y)
        correlations.append({'Feature': col, 'Correlation': corr})
    
    return pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)

def perform_pca_analysis(X: pd.DataFrame, n_components: int = 3) -> Tuple[np.ndarray, PCA]:
    """Perform PCA analysis on features"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=min(n_components, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca

def detect_outliers(X: pd.DataFrame, method: str = 'iqr') -> np.ndarray:
    """Detect outliers in the dataset"""
    if method == 'iqr':
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_mask = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
        return outlier_mask.values
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(X))
        outlier_mask = (z_scores > 3).any(axis=1)
        return outlier_mask
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")