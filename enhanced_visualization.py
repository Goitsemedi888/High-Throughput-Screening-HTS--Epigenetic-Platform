#=============================================================================
# enhanced_visualization.py
"""
Enhanced visualization module with advanced ML plotting capabilities
"""
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class EnhancedMLVisualizer:
    """Enhanced visualization class with advanced ML plotting capabilities"""
    
    def __init__(self, theme: str = "plotly_dark"):
        self.theme = theme
        self.color_palette = {
            'primary': '#00d4aa',
            'secondary': '#ffab00',
            'danger': '#ff5252',
            'warning': '#ff9500',
            'info': '#2196f3',
            'success': '#4caf50'
        }
    
    def create_model_comparison_plot(self, results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive model comparison visualization"""
        try:
            # Extract model performance data
            model_data = []
            for model_name, result in results.items():
                if isinstance(result, dict) and 'scores' in result:
                    scores = result['scores']
                    model_data.append({
                        'Model': model_name,
                        'R² Score': scores.get('r2_score', 0),
                        'RMSE': scores.get('rmse', float('inf')),
                        'MAE': scores.get('mae', float('inf')),
                        'CV Mean': scores.get('cv_mean', 0),
                        'CV Std': scores.get('cv_std', 0)
                    })
            
            if not model_data:
                return self._create_empty_plot("No model data available")
            
            df = pd.DataFrame(model_data)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(['R² Score Comparison', 'RMSE Comparison', 
                               'Cross-Validation Performance', 'Error Metrics']),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # R² Score comparison
            fig.add_trace(
                go.Bar(
                    x=df['Model'], y=df['R² Score'],
                    name='R² Score',
                    marker_color=self.color_palette['primary'],
                    text=df['R² Score'].round(3),
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # RMSE comparison
            fig.add_trace(
                go.Bar(
                    x=df['Model'], y=df['RMSE'],
                    name='RMSE',
                    marker_color=self.color_palette['danger'],
                    text=df['RMSE'].round(3),
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Cross-validation performance with error bars
            fig.add_trace(
                go.Bar(
                    x=df['Model'], y=df['CV Mean'],
                    error_y=dict(type='data', array=df['CV Std']),
                    name='CV Performance',
                    marker_color=self.color_palette['info'],
                    text=df['CV Mean'].round(3),
                    textposition='auto'
                ),
                row=2, col=1
            )
            
            # MAE comparison
            fig.add_trace(
                go.Bar(
                    x=df['Model'], y=df['MAE'],
                    name='MAE',
                    marker_color=self.color_palette['warning'],
                    text=df['MAE'].round(3),
                    textposition='auto'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="🔍 Comprehensive Model Performance Comparison",
                template=self.theme,
                showlegend=False,
                height=700,
                font=dict(color='white')
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Model comparison plot creation failed: {str(e)}")
            return self._create_error_plot(f"Model comparison failed: {str(e)}")
    
    def create_shap_summary_plot(self, shap_values: np.ndarray, 
                                feature_names: List[str],
                                X_data: pd.DataFrame) -> go.Figure:
        """Create SHAP summary plot using Plotly"""
        try:
            if shap_values is None or len(shap_values) == 0:
                return self._create_empty_plot("No SHAP values available")
            
            # Handle multi-dimensional SHAP values
            if len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 0]
            
            # Calculate feature importance (mean absolute SHAP values)
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Sort features by importance
            sorted_indices = np.argsort(feature_importance)[::-1]
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_importance = feature_importance[sorted_indices]
            
            # Create SHAP summary plot
            fig = go.Figure()
            
            # Add horizontal bar chart for feature importance
            fig.add_trace(
                go.Bar(
                    x=sorted_importance[:15],  # Top 15 features
                    y=sorted_features[:15],
                    orientation='h',
                    marker=dict(
                        color=sorted_importance[:15],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Mean |SHAP value|")
                    ),
                    text=[f"{val:.3f}" for val in sorted_importance[:15]],
                    textposition='auto'
                )
            )
            
            fig.update_layout(
                title="🧠 SHAP Feature Importance Summary",
                xaxis_title="Mean |SHAP value| (average impact on model output)",
                yaxis_title="Features",
                template=self.theme,
                height=600,
                font=dict(color='white')
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"SHAP summary plot creation failed: {str(e)}")
            return self._create_error_plot(f"SHAP plot failed: {str(e)}")
    
    def create_shap_waterfall_plot(self, shap_values: np.ndarray, 
                                  feature_names: List[str],
                                  X_data: pd.DataFrame,
                                  instance_idx: int = 0) -> go.Figure:
        """Create SHAP waterfall plot for a specific prediction"""
        try:
            if shap_values is None or len(shap_values) == 0:
                return self._create_empty_plot("No SHAP values available")
            
            # Handle multi-dimensional SHAP values
            if len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 0]
            
            # Get SHAP values for specific instance
            instance_shap = shap_values[instance_idx]
            instance_features = X_data.iloc[instance_idx]
            
            # Sort by absolute SHAP value
            sorted_indices = np.argsort(np.abs(instance_shap))[::-1]
            top_n = min(10, len(sorted_indices))  # Top 10 features
            
            # Prepare data for waterfall
            feature_contributions = []
            cumulative = 0
            
            for i in range(top_n):
                idx = sorted_indices[i]
                shap_val = instance_shap[idx]
                feature_name = feature_names[idx]
                feature_val = instance_features.iloc[idx]
                
                feature_contributions.append({
                    'feature': f"{feature_name}={feature_val:.2f}",
                    'shap_value': shap_val,
                    'cumulative': cumulative + shap_val,
                    'color': self.color_palette['success'] if shap_val > 0 else self.color_palette['danger']
                })
                cumulative += shap_val
            
            # Create waterfall plot
            fig = go.Figure()
            
            x_labels = [contrib['feature'] for contrib in feature_contributions]
            shap_vals = [contrib['shap_value'] for contrib in feature_contributions]
            colors = [contrib['color'] for contrib in feature_contributions]
            
            fig.add_trace(
                go.Waterfall(
                    name="SHAP Values",
                    orientation="v",
                    measure=["relative"] * len(x_labels),
                    x=x_labels,
                    textposition="auto",
                    text=[f"{val:+.3f}" for val in shap_vals],
                    y=shap_vals,
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": self.color_palette['success']}},
                    decreasing={"marker": {"color": self.color_palette['danger']}}
                )
            )
            
            fig.update_layout(
                title=f"🌊 SHAP Waterfall Plot - Instance {instance_idx}",
                xaxis_title="Features (with values)",
                yaxis_title="SHAP Value",
                template=self.theme,
                height=500,
                font=dict(color='white'),
                xaxis_tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"SHAP waterfall plot creation failed: {str(e)}")
            return self._create_error_plot(f"SHAP waterfall failed: {str(e)}")
    
    def create_learning_curve_plot(self, train_sizes: np.ndarray,
                                  train_scores: np.ndarray,
                                  val_scores: np.ndarray,
                                  model_name: str = "Model") -> go.Figure:
        """Create learning curve visualization"""
        try:
            # Calculate statistics
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            fig = go.Figure()
            
            # Training scores
            fig.add_trace(
                go.Scatter(
                    x=train_sizes, y=train_mean,
                    mode='lines+markers',
                    name='Training Score',
                    line=dict(color=self.color_palette['primary'], width=3),
                    error_y=dict(type='data', array=train_std, visible=True)
                )
            )
            
            # Validation scores
            fig.add_trace(
                go.Scatter(
                    x=train_sizes, y=val_mean,
                    mode='lines+markers',
                    name='Validation Score',
                    line=dict(color=self.color_palette['danger'], width=3),
                    error_y=dict(type='data', array=val_std, visible=True)
                )
            )
            
            fig.update_layout(
                title=f"📈 Learning Curve - {model_name}",
                xaxis_title="Training Set Size",
                yaxis_title="Score (R²)",
                template=self.theme,
                height=500,
                font=dict(color='white'),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Learning curve plot creation failed: {str(e)}")
            return self._create_error_plot(f"Learning curve failed: {str(e)}")
    
    def create_residuals_analysis_plot(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     model_name: str = "Model") -> go.Figure:
        """Create comprehensive residuals analysis plot"""
        try:
            residuals = y_true - y_pred
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Residuals vs Predicted', 'Residuals Distribution', 
                               'Q-Q Plot', 'Residuals vs True Values'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Residuals vs Predicted
            fig.add_trace(
                go.Scatter(
                    x=y_pred, y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color=self.color_palette['primary'], opacity=0.7),
                    showlegend=False
                ),
                row=1, col=1
            )
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color=self.color_palette['warning'], 
                         row=1, col=1)
            
            # 2. Residuals Distribution
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    name='Distribution',
                    marker_color=self.color_palette['info'],
                    opacity=0.7,
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # 3. Q-Q Plot
            from scipy import stats
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            sample_quantiles = np.sort(residuals)
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles, y=sample_quantiles,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color=self.color_palette['secondary'], opacity=0.7),
                    showlegend=False
                ),
                row=2, col=1
            )
            # Add diagonal line
            min_val = min(min(theoretical_quantiles), min(sample_quantiles))
            max_val = max(max(theoretical_quantiles), max(sample_quantiles))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    line=dict(color=self.color_palette['warning'], dash='dash'),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 4. Residuals vs True Values
            fig.add_trace(
                go.Scatter(
                    x=y_true, y=residuals,
                    mode='markers',
                    name='Residuals vs True',
                    marker=dict(color=self.color_palette['success'], opacity=0.7),
                    showlegend=False
                ),
                row=2, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color=self.color_palette['warning'],
                         row=2, col=2)
            
            fig.update_layout(
                title_text=f"🔍 Residuals Analysis - {model_name}",
                template=self.theme,
                height=700,
                font=dict(color='white')
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Residuals analysis plot creation failed: {str(e)}")
            return self._create_error_plot(f"Residuals analysis failed: {str(e)}")
    
    def create_feature_correlation_network(self, correlation_matrix: pd.DataFrame,
                                         threshold: float = 0.5) -> go.Figure:
        """Create feature correlation network visualization"""
        try:
            import networkx as nx
            
            # Create network from correlation matrix
            G = nx.Graph()
            features = correlation_matrix.columns.tolist()
            
            # Add nodes
            for feature in features:
                G.add_node(feature)
            
            # Add edges for significant correlations
            for i, feature1 in enumerate(features):
                for j, feature2 in enumerate(features):
                    if i < j:  # Avoid duplicates
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) >= threshold:
                            G.add_edge(feature1, feature2, weight=abs(corr_val), 
                                     correlation=corr_val)
            
            # Calculate layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Prepare edge traces
            edge_x = []
            edge_y = []
            edge_colors = []
            edge_widths = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                corr_val = G[edge[0]][edge[1]]['correlation']
                edge_colors.append('red' if corr_val < 0 else 'blue')
                edge_widths.append(abs(corr_val) * 5)
            
            # Prepare node traces
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_text = list(G.nodes())
            node_degrees = [G.degree(node) for node in G.nodes()]
            
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=0.5, color='rgba(125,125,125,0.5)'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
            
            # Add nodes
            fig.add_trace(
                go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    marker=dict(
                        size=[degree * 5 + 10 for degree in node_degrees],
                        color=node_degrees,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Node Degree")
                    ),
                    text=node_text,
                    textposition="middle center",
                    hoverinfo='text',
                    showlegend=False
                )
            )
            
            fig.update_layout(
                title="🕸️ Feature Correlation Network",
                template=self.theme,
                height=600,
                font=dict(color='white'),
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                annotations=[
                    dict(
                        text=f"Correlation threshold: {threshold}<br>"
                             f"Red edges: negative correlation<br>"
                             f"Blue edges: positive correlation",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.02, y=0.98, xanchor="left", yanchor="top",
                        align="left",
                        bgcolor="rgba(0,0,0,0.8)",
                        bordercolor="white",
                        borderwidth=1
                    )
                ]
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Correlation network plot creation failed: {str(e)}")
            return self._create_error_plot(f"Correlation network failed: {str(e)}")
    
    def create_prediction_confidence_plot(self, y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        prediction_std: Optional[np.ndarray] = None,
                                        model_name: str = "Model") -> go.Figure:
        """Create prediction vs actual with confidence intervals"""
        try:
            fig = go.Figure()
            
            # Perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color=self.color_palette['warning'], dash='dash', width=2)
                )
            )
            
            # Prediction points
            if prediction_std is not None:
                # With confidence intervals
                fig.add_trace(
                    go.Scatter(
                        x=y_true, y=y_pred,
                        mode='markers',
                        name='Predictions',
                        marker=dict(
                            color=prediction_std,
                            colorscale='Viridis',
                            size=8,
                            showscale=True,
                            colorbar=dict(title="Prediction Uncertainty")
                        ),
                        error_y=dict(
                            type='data',
                            array=prediction_std,
                            visible=True
                        ),
                        hovertemplate='True: %{x:.3f}<br>Predicted: %{y:.3f}<br>Std: %{marker.color:.3f}<extra></extra>'
                    )
                )
            else:
                # Without confidence intervals
                fig.add_trace(
                    go.Scatter(
                        x=y_true, y=y_pred,
                        mode='markers',
                        name='Predictions',
                        marker=dict(
                            color=self.color_palette['primary'],
                            size=8,
                            opacity=0.7
                        ),
                        hovertemplate='True: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>'
                    )
                )
            
            # Calculate R²
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            
            fig.update_layout(
                title=f"🎯 Predictions vs Actual - {model_name} (R² = {r2:.3f})",
                xaxis_title="True Values",
                yaxis_title="Predicted Values",
                template=self.theme,
                height=500,
                font=dict(color='white'),
                annotations=[
                    dict(
                        text=f"R² = {r2:.3f}<br>"
                             f"RMSE = {np.sqrt(np.mean((y_true - y_pred)**2)):.3f}",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.02, y=0.98, xanchor="left", yanchor="top",
                        bgcolor="rgba(0,0,0,0.8)",
                        bordercolor="white",
                        borderwidth=1
                    )
                ]
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Prediction confidence plot creation failed: {str(e)}")
            return self._create_error_plot(f"Prediction confidence failed: {str(e)}")
    
    def create_hyperparameter_optimization_plot(self, param_history: List[Dict],
                                              param_name: str,
                                              score_name: str = 'score') -> go.Figure:
        """Create hyperparameter optimization history plot"""
        try:
            if not param_history:
                return self._create_empty_plot("No hyperparameter optimization history")
            
            # Extract data
            param_values = [entry[param_name] for entry in param_history if param_name in entry]
            scores = [entry[score_name] for entry in param_history if score_name in entry]
            iterations = list(range(len(param_values)))
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[f'{param_name} vs Iteration', f'{score_name} vs {param_name}'],
                vertical_spacing=0.1
            )
            
            # Parameter evolution over iterations
            fig.add_trace(
                go.Scatter(
                    x=iterations, y=param_values,
                    mode='lines+markers',
                    name=f'{param_name} Evolution',
                    line=dict(color=self.color_palette['primary'], width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            # Score vs parameter value
            fig.add_trace(
                go.Scatter(
                    x=param_values, y=scores,
                    mode='markers',
                    name=f'{score_name} vs {param_name}',
                    marker=dict(
                        color=scores,
                        colorscale='Viridis',
                        size=8,
                        showscale=True,
                        colorbar=dict(title=score_name, y=0.3, len=0.4)
                    )
                ),
                row=2, col=1
            )
            
            # Highlight best parameter
            best_idx = np.argmax(scores)
            best_param = param_values[best_idx]
            best_score = scores[best_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=[best_param], y=[best_score],
                    mode='markers',
                    name=f'Best: {param_name}={best_param}',
                    marker=dict(color=self.color_palette['success'], size=12, symbol='star'),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title_text=f"⚙️ Hyperparameter Optimization - {param_name}",
                template=self.theme,
                height=600,
                font=dict(color='white')
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization plot creation failed: {str(e)}")
            return self._create_error_plot(f"Hyperparameter plot failed: {str(e)}")
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color='white')
        )
        fig.update_layout(
            template=self.theme,
            height=400,
            font=dict(color='white'),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
    
    def _create_error_plot(self, error_message: str) -> go.Figure:
        """Create an error plot"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color=self.color_palette['danger'])
        )
        fig.update_layout(
            template=self.theme,
            height=400,
            font=dict(color='white'),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig

# NEW: Enhanced ML Analysis Tab with both SHAP and LIME explanations
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
            enable_explanations: Whether to enable SHAP/LIME explanations
        """
        st.markdown("### 🤖 Advanced Machine Learning Analysis")
        
        if data is None or len(data) == 0:
            st.warning("No data available for ML analysis.")
            return
        
        # Check for required features
        feature_cols = [col for col in data.columns if col in [
            'MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'AromaticRings', 
            'Complexity', 'SlogP', 'NumHeavyAtoms', 'FractionCsp3', 'NumAliphaticRings'
        ]]
        
        if len(feature_cols) == 0:
            st.error("❌ No chemical descriptors found for ML analysis. Required features: MW, LogP, HBD, HBA, etc.")
            return
        
        try:
            # Prepare data
            X, y, feature_names = self._prepare_ml_data(data, feature_cols)
            
            if X is None or len(X) == 0:
                st.error("❌ Insufficient data for ML analysis")
                return
            
            # UI Controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                models_to_run = st.multiselect(
                    "🎯 Select Models",
                    ['Random Forest', 'Gradient Boosting', 'Ridge Regression', 'Support Vector Regression'],
                    default=['Random Forest', 'Gradient Boosting']
                )
            
            with col2:
                test_size = st.slider("📊 Test Set Size", 0.1, 0.5, 0.2, 0.05)
                enable_hyperopt = st.checkbox("⚙️ Hyperparameter Tuning", value=True)
            
            with col3:
                explanation_method = st.selectbox(
                    "🔍 Explanation Method",
                    ["None", "SHAP", "LIME", "Both SHAP & LIME"],
                    index=3 if enable_explanations else 0
                )
                cross_val_folds = st.slider("📈 Cross-Validation Folds", 3, 10, 5)
            
            # Run analysis button
            if st.button("🚀 Run ML Analysis", type="primary"):
                with st.spinner("🔄 Training models and generating explanations..."):
                    results = self._run_ml_analysis(
                        X, y, feature_names, models_to_run, test_size, 
                        enable_hyperopt, explanation_method, cross_val_folds
                    )
                    
                    if results:
                        self._display_results(results, explanation_method)
                    else:
                        st.error("❌ ML analysis failed. Check the logs for details.")
        
        except Exception as e:
            st.error(f"❌ ML Analysis Error: {str(e)}")
            logger.error(f"Enhanced ML tab error: {str(e)}\n{traceback.format_exc()}")
    
    def _prepare_ml_data(self, data: pd.DataFrame, feature_cols: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
        """Prepare data for ML analysis"""
        try:
            # Filter for test compounds only (exclude controls)
            test_compounds = data[~data['Compound'].str.contains('Control|DMSO', case=False, na=False)]
            
            if len(test_compounds) < 10:
                st.warning("⚠️ Insufficient test compounds for reliable ML analysis (need ≥10)")
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
                st.info(f"📋 Removed constant features: {constant_features}")
            
            # Remove highly correlated features (>0.95)
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            
            if to_drop:
                X = X.drop(columns=to_drop)
                st.info(f"📋 Removed highly correlated features: {to_drop}")
            
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
            st.warning("⚠️ SHAP not installed. Run: pip install shap")
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
            st.warning("⚠️ LIME not installed. Run: pip install lime")
            return None
        except Exception as e:
            logger.warning(f"LIME explanation failed for {model_name}: {str(e)}")
            return None
    
    def _display_results(self, results: Dict[str, Any], explanation_method: str):
        """Display ML analysis results"""
        try:
            models = results['models']
            if not models:
                st.error("❌ No models were successfully trained")
                return
            
            # Model comparison
            st.markdown("### 📊 Model Performance Comparison")
            
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
            st.success(f"🏆 Best Model: **{best_model_name}** (R² = {comparison_df['Test R²'].max():.3f})")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # R² comparison
                fig_r2 = px.bar(
                    comparison_df, x='Model', y='Test R²',
                    title='🎯 Test R² Comparison',
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
                    title='📈 Cross-Validation Performance',
                    xaxis_title='Model',
                    yaxis_title='CV R²',
                    template='plotly_dark',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_cv, use_container_width=True)
            
            # Detailed analysis for best model
            st.markdown(f"### 🔍 Detailed Analysis - {best_model_name}")
            
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
                title=f'🎯 Predictions vs Actual - {best_model_name}',
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
                st.markdown("### 🧠 Model Explanations")
                
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
                        shap_tab, lime_tab = st.tabs(["🎯 SHAP Analysis", "🔍 LIME Analysis"])
                        
                        with shap_tab:
                            self._display_shap_explanations(best_result, metadata)
                        
                        with lime_tab:
                            self._display_lime_explanations(best_result)
            
            # Hyperparameter information
            if best_result['best_params']:
                st.markdown("### ⚙️ Best Hyperparameters")
                st.json(best_result['best_params'])
            
        except Exception as e:
            st.error(f"❌ Results display failed: {str(e)}")
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
                title='🎯 Top 10 Feature Importances',
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
                st.warning("⚠️ No SHAP values available")
                return
            
            # Handle multi-dimensional SHAP values
            if len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 0]
            
            st.markdown("#### 🎯 SHAP Feature Importance")
            
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
                st.markdown("#### 🌊 SHAP Waterfall Plot (First Instance)")
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
            st.error(f"❌ SHAP display failed: {str(e)}")
            logger.error(f"SHAP display error: {str(e)}")
    
    def _display_lime_explanations(self, result: Dict[str, Any]):
        """Display LIME explanations"""
        try:
            lime_explanations = result['lime_explanations']
            
            if lime_explanations is None:
                st.warning("⚠️ No LIME explanations available")
                return
            
            st.markdown("#### 🔍 LIME Local Explanations")
            
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
            st.markdown("#### 📊 LIME vs SHAP Comparison")
            st.info("""
            **LIME vs SHAP:**
            - **LIME**: Local explanations, faster, model-agnostic, explains individual predictions
            - **SHAP**: Global + local explanations, theoretically grounded, unified framework
            - **Use Case**: LIME for quick local insights, SHAP for comprehensive analysis
            """)
            
        except Exception as e:
            st.error(f"❌ LIME display failed: {str(e)}")
            logger.error(f"LIME display error: {str(e)}")

# Integration with existing visualizer
class ProductionHTSVisualizer:
    """Production-ready HTS visualizer with enhanced ML capabilities"""

    def __init__(self):
        self.ml_visualizer = EnhancedMLVisualizer()
        self.ml_analysis_tab = EnhancedMLAnalysisTab()  # NEW: Add ML analysis tab
        self.theme = "plotly_dark"
        self.color_palette = {
            'primary': '#00d4aa',
            'secondary': '#ffab00',
            'danger': '#ff5252',
            'warning': '#ff9500'
        }

    def create_hits_barplot(self, hits: pd.DataFrame, hit_threshold: float) -> go.Figure:
        """Create a bar plot of top hits"""
        try:
            if hits.empty:
                return self._create_empty_plot("No hits found")
            
            # Sort hits by activity and take top 15
            top_hits = hits.nlargest(15, 'Expression_Fold_Change').copy()
            
            # Create activity categories
            top_hits['Activity_Category'] = pd.cut(
                top_hits['Expression_Fold_Change'], 
                bins=[hit_threshold, 2.0, 3.0, float('inf')], 
                labels=['Moderate', 'Strong', 'Very Strong'],
                right=False
            )
            
            fig = px.bar(
                top_hits, x="Well", y="Expression_Fold_Change", 
                color="Activity_Category", 
                title="🏆 Top 15 Hit Compounds",
                template=self.theme, 
                hover_data=['Compound', 'Expression_Fold_Change']
            )
            
            # Add hit threshold line
            fig.add_hline(
                y=hit_threshold, 
                line_dash="dash", 
                line_color=self.color_palette['warning'],
                annotation_text=f"Hit Threshold ({hit_threshold})"
            )
            
            fig.update_xaxes(tickangle=45)
            fig.update_layout(
                xaxis_title="Well Position",
                yaxis_title="Expression Fold Change",
                height=500,
                font=dict(color='white')
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Hits barplot creation failed: {str(e)}")
            return self._create_error_plot(f"Hits visualization failed: {str(e)}")

    # Add this helper method if it doesn't exist
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color='white')
        )
        fig.update_layout(
            template=self.theme,
            height=400,
            font=dict(color='white'),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig

    def _create_error_plot(self, error_message: str) -> go.Figure:
        """Create an error plot"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color=self.color_palette['danger'])
        )
        fig.update_layout(
            template=self.theme,
            height=400,
            font=dict(color='white'),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
    
    def render_plate_analysis(self, data: pd.DataFrame, hit_threshold: float):
        """Render plate heatmap, hit statistics, and QC metrics"""
        st.markdown("### 🧪 Plate Analysis")

        if data is None or data.empty:
            st.warning("No data available for plate analysis.")
            return

        # Plate heatmap
        if {"Well", "Expression_Fold_Change"}.issubset(data.columns):
            try:
                df = data.copy()
                df["Row"] = df["Well"].str[0]
                df["Column"] = df["Well"].str[1:].astype(int)
                heatmap_data = df.pivot(index="Row", columns="Column", values="Expression_Fold_Change")
                fig = px.imshow(
                    heatmap_data,
                    color_continuous_scale="Viridis",
                    title="Plate Heatmap (Expression Fold Change)"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Heatmap generation failed: {str(e)}")

        # Basic hit stats + QC metrics
        if "Expression_Fold_Change" in data.columns:
            hits = data[data["Expression_Fold_Change"] >= hit_threshold]
            st.metric("Hit Count", len(hits))
            st.metric("Hit Rate (%)", f"{len(hits) / len(data) * 100:.1f}")

            try:
                values = data["Expression_Fold_Change"].dropna()
                mean_signal = values.mean()
                mean_background = values.min()
                std_signal = values.std(ddof=1)
                std_background = values.std(ddof=1)
                # Z'-factor
                z_factor = 1 - (3 * (std_signal + std_background)) / abs(mean_signal - mean_background)
                # CV (%)
                cv = 100 * std_signal / mean_signal if mean_signal != 0 else np.nan
                # Signal/background ratio
                s_b_ratio = mean_signal / mean_background if mean_background != 0 else np.nan

                st.markdown("#### 📊 Assay Quality Metrics")
                st.metric("Z'-Factor", f"{z_factor:.3f}")
                st.metric("Signal / Background", f"{s_b_ratio:.2f}")
                st.metric("CV (%)", f"{cv:.1f}")
            except Exception as e:
                st.error(f"QC metric calculation failed: {str(e)}")

    def render_quality_control(self, data: pd.DataFrame, metrics: dict, confidence_level: float = 0.95):
        """Render assay quality control metrics with interpretation"""
        st.markdown("### 📊 Assay Quality Control")

        if data is None or data.empty:
            st.warning("No data available for quality control.")
            return

        try:
            # Show main QC metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Z'-Factor", f"{metrics.get('z_factor', np.nan):.3f}")
            with col2:
                st.metric("Signal/Background", f"{metrics.get('signal_background', np.nan):.2f}")
            with col3:
                st.metric("CV (%)", f"{metrics.get('cv', np.nan):.1f}")

            # Interpretation
            st.markdown("#### 📝 Interpretation")
            z = metrics.get("z_factor", 0)
            if z >= 0.5:
                st.success("✅ Excellent assay quality (Z'-factor ≥ 0.5)")
            elif 0 < z < 0.5:
                st.warning("⚠️ Marginal assay quality (0 < Z'-factor < 0.5)")
            else:
                st.error("❌ Poor assay quality (Z'-factor ≤ 0)")

            # Confidence intervals
            if "Expression_Fold_Change" in data.columns:
                import scipy.stats as stats
                values = data["Expression_Fold_Change"].dropna()
                mean = values.mean()
                sem = stats.sem(values)
                ci = stats.t.interval(confidence_level, len(values)-1, loc=mean, scale=sem)

                st.markdown(f"**{int(confidence_level*100)}% Confidence Interval (Expression Fold Change):**")
                st.write(f"[{ci[0]:.3f}, {ci[1]:.3f}]")

        except Exception as e:
            st.error(f"❌ QC rendering failed: {str(e)}")

    def create_comprehensive_ml_report(self, ml_results: dict) -> dict:
        """Create comprehensive ML visualization report"""
        plots = {}
        try:
            plots['model_comparison'] = self.ml_visualizer.create_model_comparison_plot(ml_results)
            if 'metadata' in ml_results and 'best_model_name' in ml_results['metadata']:
                best_model_name = ml_results['metadata']['best_model_name']
                best_result = ml_results[best_model_name]
                if 'model' in best_result and 'X_test' in ml_results['metadata']:
                    y_true = ml_results['metadata']['y_test']
                    y_pred = best_result.predictions
                    plots['residuals_analysis'] = self.ml_visualizer.create_residuals_analysis_plot(
                        y_true, y_pred, best_model_name
                    )
                if hasattr(best_result, 'shap_values') and best_result.shap_values is not None:
                    plots['shap_summary'] = self.ml_visualizer.create_shap_summary_plot(
                        best_result.shap_values,
                        ml_results['metadata']['feature_names'],
                        ml_results['metadata']['X_test']
                    )
                    plots['shap_waterfall'] = self.ml_visualizer.create_shap_waterfall_plot(
                        best_result.shap_values,
                        ml_results['metadata']['feature_names'],
                        ml_results['metadata']['X_test']
                    )
        except Exception as e:
            logging.getLogger(__name__).error(f"Comprehensive ML report creation failed: {str(e)}")
        return plots

    def render_ml_analysis_tab(self, data: pd.DataFrame, hit_threshold: float, enable_explanations: bool = True):
        """Render the Enhanced ML Analysis tab"""
        self.ml_analysis_tab.render(data, hit_threshold, enable_explanations)

    def create_comprehensive_ml_report(self, ml_results: dict) -> dict:
        """Create comprehensive ML visualization report"""
        plots = {}
        try:
            plots['model_comparison'] = self.ml_visualizer.create_model_comparison_plot(ml_results)
            if 'metadata' in ml_results and 'best_model_name' in ml_results['metadata']:
                best_model_name = ml_results['metadata']['best_model_name']
                best_result = ml_results[best_model_name]
                if 'model' in best_result and 'X_test' in ml_results['metadata']:
                    y_true = ml_results['metadata']['y_test']
                    y_pred = best_result.predictions
                    plots['residuals_analysis'] = self.ml_visualizer.create_residuals_analysis_plot(
                        y_true, y_pred, best_model_name
                    )
                if hasattr(best_result, 'shap_values') and best_result.shap_values is not None:
                    plots['shap_summary'] = self.ml_visualizer.create_shap_summary_plot(
                        best_result.shap_values,
                        ml_results['metadata']['feature_names'],
                        ml_results['metadata']['X_test']
                    )
                    plots['shap_waterfall'] = self.ml_visualizer.create_shap_waterfall_plot(
                        best_result.shap_values,
                        ml_results['metadata']['feature_names'],
                        ml_results['metadata']['X_test']
                    )
        except Exception as e:
            logging.getLogger(__name__).error(f"Comprehensive ML report creation failed: {str(e)}")
        return plots

    # NEW: Method to render the ML analysis tab
    def render_ml_analysis_tab(self, data: pd.DataFrame, hit_threshold: float, enable_explanations: bool = True):
        """Render the Enhanced ML Analysis tab"""
        self.ml_analysis_tab.render(data, hit_threshold, enable_explanations)

    # [Your existing methods remain unchanged]
    # render_plate_analysis, render_quality_control

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    
    # Sample ML results for testing
    sample_results = {
        'Random Forest': {
            'scores': {
                'r2_score': 0.85,
                'rmse': 0.25,
                'mae': 0.18,
                'cv_mean': 0.82,
                'cv_std': 0.03
            }
        },
        'Gradient Boosting': {
            'scores': {
                'r2_score': 0.88,
                'rmse': 0.22,
                'mae': 0.16,
                'cv_mean': 0.86,
                'cv_std': 0.02
            }
        }
    }
    
    visualizer = ProductionHTSVisualizer()
    
    # Test model comparison plot
    comparison_plot = visualizer.ml_visualizer.create_model_comparison_plot(sample_results)
    print("Model comparison plot created successfully!")
    
    # Test with sample SHAP data
    sample_shap = np.random.randn(100, 8)
    feature_names = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'AromaticRings', 'Complexity']
    sample_X = pd.DataFrame(np.random.randn(100, 8), columns=feature_names)
    
    shap_plot = visualizer.ml_visualizer.create_shap_summary_plot(
        sample_shap, feature_names, sample_X
    )
    print("SHAP summary plot created successfully!")
    
    # Test ML analysis tab
    sample_data = pd.DataFrame({
        'Compound': [f'Test_{i}' for i in range(100)],
        'MW': np.random.normal(300, 50, 100),
        'LogP': np.random.normal(3, 1, 100),
        'HBD': np.random.randint(0, 5, 100),
        'HBA': np.random.randint(0, 10, 100),
        'TPSA': np.random.normal(80, 20, 100),
        'Expression_Fold_Change': np.random.normal(2, 0.5, 100)
    })
    
    ml_tab = EnhancedMLAnalysisTab()
    print("Enhanced ML Analysis Tab created successfully!")
    