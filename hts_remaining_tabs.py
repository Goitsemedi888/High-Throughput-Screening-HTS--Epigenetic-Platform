#=============================================================================
# hts_remaining_tabs.py - Fixed Version
"""
Remaining tabs for the HTS Dashboard with proper parameter handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import logging
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class HitIdentificationTab:
    """Hit Identification Tab with enhanced analysis"""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.color_palette = {
            'primary': '#00d4aa',
            'secondary': '#ffab00',
            'danger': '#ff5252',
            'warning': '#ff9500',
            'info': '#2196f3',
            'success': '#4caf50'
        }
    
    def render(self, data: pd.DataFrame, hit_threshold: float):
        """
        Render hit identification analysis
        
        Args:
            data: HTS screening data
            hit_threshold: Threshold for hit identification
        """
        if data is None or len(data) == 0:
            st.warning("No data available for hit identification.")
            return
        
        try:
            st.markdown("### Hit Identification & Prioritization")
            
            # Identify hits
            hits = data[data['Expression_Fold_Change'] >= hit_threshold]
            
            if len(hits) == 0:
                st.warning(f"No hits found with threshold {hit_threshold:.1f}")
                return
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Hits", len(hits))
            
            with col2:
                hit_rate = len(hits) / len(data) * 100
                st.metric("Hit Rate", f"{hit_rate:.1f}%")
            
            with col3:
                if len(hits) > 0:
                    avg_activity = hits['Expression_Fold_Change'].mean()
                    st.metric("Avg Hit Activity", f"{avg_activity:.2f}")
                else:
                    st.metric("Avg Hit Activity", "N/A")
            
            with col4:
                if len(hits) > 0:
                    max_activity = hits['Expression_Fold_Change'].max()
                    st.metric("Max Activity", f"{max_activity:.2f}")
                else:
                    st.metric("Max Activity", "N/A")
            
            # Hit distribution plot
            if len(hits) > 0:
                # Create hits bar plot
                hits_plot = self.visualizer.create_hits_barplot(hits, hit_threshold)
                st.plotly_chart(hits_plot, use_container_width=True)
                
                # Hit analysis tabs
                hit_tab1, hit_tab2, hit_tab3 = st.tabs(["Top Hits", "Hit Distribution", "Activity Analysis"])
                
                with hit_tab1:
                    self._display_top_hits(hits)
                
                with hit_tab2:
                    self._display_hit_distribution(hits, hit_threshold)
                
                with hit_tab3:
                    self._display_activity_analysis(data, hits, hit_threshold)
            
        except Exception as e:
            st.error(f"Error in hit identification: {str(e)}")
            logger.error(f"Hit identification error: {str(e)}\n{traceback.format_exc()}")
    
    def _display_top_hits(self, hits: pd.DataFrame):
        """Display top hits table"""
        try:
            st.markdown("#### Top Hit Compounds")
            
            # Sort hits by activity
            top_hits = hits.nlargest(20, 'Expression_Fold_Change')
            
            # Display table
            display_cols = ['Well', 'Compound', 'Expression_Fold_Change']
            if 'MW' in hits.columns:
                display_cols.extend(['MW', 'LogP', 'HBD', 'HBA'])
            
            available_cols = [col for col in display_cols if col in top_hits.columns]
            st.dataframe(
                top_hits[available_cols].round(3),
                use_container_width=True,
                hide_index=True
            )
            
        except Exception as e:
            st.error(f"Error displaying top hits: {str(e)}")
    
    def _display_hit_distribution(self, hits: pd.DataFrame, hit_threshold: float):
        """Display hit distribution analysis"""
        try:
            st.markdown("#### Hit Activity Distribution")
            
            # Activity histogram
            fig = px.histogram(
                hits, x='Expression_Fold_Change',
                nbins=20, title='Hit Activity Distribution',
                color_discrete_sequence=[self.color_palette['primary']]
            )
            
            fig.add_vline(
                x=hit_threshold, line_dash="dash", 
                line_color=self.color_palette['warning'],
                annotation_text=f"Hit Threshold ({hit_threshold})"
            )
            
            fig.update_layout(template='plotly_dark', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Activity categories
            if len(hits) > 0:
                categories = pd.cut(
                    hits['Expression_Fold_Change'],
                    bins=[hit_threshold, 2.0, 3.0, float('inf')],
                    labels=['Moderate (1.5-2x)', 'Strong (2-3x)', 'Very Strong (>3x)'],
                    right=False
                )
                
                category_counts = categories.value_counts()
                
                fig_pie = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title='Hit Activity Categories',
                    color_discrete_sequence=[
                        self.color_palette['info'], 
                        self.color_palette['primary'], 
                        self.color_palette['success']
                    ]
                )
                fig_pie.update_layout(template='plotly_dark', font=dict(color='white'))
                st.plotly_chart(fig_pie, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in hit distribution analysis: {str(e)}")
    
    def _display_activity_analysis(self, data: pd.DataFrame, hits: pd.DataFrame, hit_threshold: float):
        """Display activity analysis"""
        try:
            st.markdown("#### Activity Analysis")
            
            # Overall vs hits comparison
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot comparison
                all_data_label = ['All Compounds'] * len(data)
                hits_label = ['Hits'] * len(hits)
                
                comparison_data = pd.DataFrame({
                    'Activity': list(data['Expression_Fold_Change']) + list(hits['Expression_Fold_Change']),
                    'Type': all_data_label + hits_label
                })
                
                fig_box = px.box(
                    comparison_data, x='Type', y='Activity',
                    title='Activity Distribution Comparison',
                    color='Type',
                    color_discrete_sequence=[self.color_palette['info'], self.color_palette['primary']]
                )
                fig_box.update_layout(template='plotly_dark', font=dict(color='white'))
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # Statistics comparison
                st.markdown("**Statistical Summary**")
                
                all_stats = data['Expression_Fold_Change'].describe()
                hits_stats = hits['Expression_Fold_Change'].describe()
                
                stats_df = pd.DataFrame({
                    'All Compounds': all_stats,
                    'Hits Only': hits_stats
                }).round(3)
                
                st.dataframe(stats_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in activity analysis: {str(e)}")

class MLAnalysisTab:
    """Machine Learning Analysis Tab"""
    
    def __init__(self, ml_pipeline, ml_visualizer):
        self.ml_pipeline = ml_pipeline
        self.ml_visualizer = ml_visualizer
        self.models = {}
        self.scaler = StandardScaler()
    
    def render(self, data: pd.DataFrame, hit_threshold: float, enable_shap: bool = True):
        """Render ML analysis tab"""
        if data is None or len(data) == 0:
            st.warning("No data available for ML analysis.")
            return
        
        try:
            st.markdown("### Machine Learning Analysis")
            
            # Check for required features
            feature_cols = [col for col in data.columns if col in [
                'MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'AromaticRings', 
                'Complexity', 'SlogP', 'NumHeavyAtoms'
            ]]
            
            if len(feature_cols) == 0:
                st.error("No chemical descriptors found for ML analysis. Required features: MW, LogP, HBD, HBA, etc.")
                return
            
            # Filter for test compounds
            test_compounds = data[~data['Compound'].str.contains('Control|DMSO', case=False, na=False)]
            
            if len(test_compounds) < 10:
                st.warning("Insufficient test compounds for ML analysis (need ≥10)")
                return
            
            # ML controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                models_to_run = st.multiselect(
                    "Select Models",
                    ['Random Forest', 'Gradient Boosting', 'Ridge Regression', 'Support Vector Regression'],
                    default=['Random Forest', 'Gradient Boosting']
                )
            
            with col2:
                test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
                enable_hyperopt = st.checkbox("Hyperparameter Tuning", value=False)
            
            with col3:
                cross_val_folds = st.slider("CV Folds", 3, 10, 5)
            
            if st.button("Run ML Analysis", type="primary"):
                with st.spinner("Training models..."):
                    results = self._run_ml_analysis(
                        test_compounds, feature_cols, models_to_run, 
                        test_size, enable_hyperopt, cross_val_folds, enable_shap
                    )
                    
                    if results:
                        self._display_ml_results(results)
        
        except Exception as e:
            st.error(f"ML Analysis Error: {str(e)}")
            logger.error(f"ML analysis error: {str(e)}\n{traceback.format_exc()}")
    
    def _run_ml_analysis(self, data: pd.DataFrame, feature_cols: List[str],
                        models_to_run: List[str], test_size: float,
                        enable_hyperopt: bool, cv_folds: int, enable_shap: bool) -> Dict:
        """Run ML analysis"""
        try:
            # Prepare data
            X = data[feature_cols].fillna(data[feature_cols].mean())
            y = data['Expression_Fold_Change'].fillna(data['Expression_Fold_Change'].mean())
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns, 
                index=X_test.index
            )
            
            results = {}
            
            # Model configurations
            model_configs = {
                'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
                'Ridge Regression': Ridge(random_state=42),
                'Support Vector Regression': SVR()
            }
            
            for model_name in models_to_run:
                if model_name not in model_configs:
                    continue
                
                model = model_configs[model_name]
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
                
                results[model_name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_test': y_test,
                    'y_pred': y_pred_test,
                    'feature_names': feature_cols
                }
            
            return results
            
        except Exception as e:
            logger.error(f"ML analysis failed: {str(e)}")
            st.error(f"ML analysis failed: {str(e)}")
            return {}
    
    def _display_ml_results(self, results: Dict):
        """Display ML results"""
        try:
            if not results:
                st.error("No ML results to display")
                return
            
            # Model comparison
            st.markdown("#### Model Performance Comparison")
            
            comparison_data = []
            for model_name, result in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Test R²': result['test_r2'],
                    'Test RMSE': result['test_rmse'],
                    'CV Mean': result['cv_mean'],
                    'CV Std': result['cv_std']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.round(4), use_container_width=True, hide_index=True)
            
            # Best model
            best_model_name = comparison_df.loc[comparison_df['Test R²'].idxmax(), 'Model']
            st.success(f"Best Model: **{best_model_name}** (R² = {comparison_df['Test R²'].max():.3f})")
            
            # Visualizations
            best_result = results[best_model_name]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Predictions vs actual
                fig = px.scatter(
                    x=best_result['y_test'], 
                    y=best_result['y_pred'],
                    title=f'Predictions vs Actual - {best_model_name}',
                    labels={'x': 'Actual', 'y': 'Predicted'}
                )
                
                # Perfect prediction line
                min_val = min(best_result['y_test'].min(), best_result['y_pred'].min())
                max_val = max(best_result['y_test'].max(), best_result['y_pred'].max())
                fig.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash")
                )
                
                fig.update_layout(template='plotly_dark', font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature importance (for tree-based models)
                if hasattr(best_result['model'], 'feature_importances_'):
                    importances = best_result['model'].feature_importances_
                    feature_names = best_result['feature_names']
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    fig_imp = px.bar(
                        importance_df.head(10),
                        x='Importance', y='Feature',
                        orientation='h',
                        title='Feature Importance'
                    )
                    fig_imp.update_layout(template='plotly_dark', font=dict(color='white'))
                    st.plotly_chart(fig_imp, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error displaying ML results: {str(e)}")

class SARAnalysisTab:
    """Structure-Activity Relationship Analysis Tab"""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
    
    def render(self, data: pd.DataFrame, hit_threshold: float):
        """Render SAR analysis"""
        if data is None or len(data) == 0:
            st.warning("No data available for SAR analysis.")
            return
        
        try:
            st.markdown("### Structure-Activity Relationship Analysis")
            
            # Check for chemical descriptors
            chem_cols = [col for col in data.columns if col in [
                'MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'AromaticRings'
            ]]
            
            if len(chem_cols) == 0:
                st.error("No chemical descriptors found for SAR analysis")
                return
            
            # Filter test compounds
            test_compounds = data[~data['Compound'].str.contains('Control|DMSO', case=False, na=False)]
            
            if len(test_compounds) < 5:
                st.warning("Insufficient test compounds for SAR analysis")
                return
            
            # SAR analysis tabs
            sar_tab1, sar_tab2, sar_tab3 = st.tabs([
                "Property vs Activity", "Correlation Analysis", "Chemical Space"
            ])
            
            with sar_tab1:
                self._property_vs_activity(test_compounds, chem_cols, hit_threshold)
            
            with sar_tab2:
                self._correlation_analysis(test_compounds, chem_cols)
            
            with sar_tab3:
                self._chemical_space_analysis(test_compounds, chem_cols)
        
        except Exception as e:
            st.error(f"SAR Analysis Error: {str(e)}")
            logger.error(f"SAR analysis error: {str(e)}\n{traceback.format_exc()}")
    
    def _property_vs_activity(self, data: pd.DataFrame, chem_cols: List[str], hit_threshold: float):
        """Property vs activity plots"""
        try:
            st.markdown("#### Chemical Properties vs Biological Activity")
            
            # Property selection
            selected_props = st.multiselect(
                "Select Properties to Analyze:",
                chem_cols,
                default=chem_cols[:4]
            )
            
            if not selected_props:
                st.info("Please select at least one property")
                return
            
            # Create scatter plots
            n_props = len(selected_props)
            n_cols = min(2, n_props)
            n_rows = (n_props + 1) // 2
            
            for i, prop in enumerate(selected_props):
                if i % 2 == 0:
                    cols = st.columns(2)
                
                col_idx = i % 2
                
                with cols[col_idx]:
                    # Color by hit status
                    data_copy = data.copy()
                    data_copy['Hit_Status'] = data_copy['Expression_Fold_Change'] >= hit_threshold
                    
                    fig = px.scatter(
                        data_copy, x=prop, y='Expression_Fold_Change',
                        color='Hit_Status',
                        title=f'{prop} vs Activity',
                        color_discrete_map={True: '#00d4aa', False: '#666666'}
                    )
                    
                    fig.add_hline(
                        y=hit_threshold, line_dash="dash",
                        line_color='orange',
                        annotation_text=f"Hit Threshold"
                    )
                    
                    fig.update_layout(template='plotly_dark', font=dict(color='white'))
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in property vs activity analysis: {str(e)}")
    
    def _correlation_analysis(self, data: pd.DataFrame, chem_cols: List[str]):
        """Correlation analysis"""
        try:
            st.markdown("#### Correlation Analysis")
            
            # Calculate correlations with activity
            correlations = []
            for col in chem_cols:
                if col in data.columns:
                    corr = data[col].corr(data['Expression_Fold_Change'])
                    correlations.append({'Property': col, 'Correlation': corr})
            
            if correlations:
                corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
                
                # Correlation bar plot
                fig = px.bar(
                    corr_df, x='Property', y='Correlation',
                    title='Property-Activity Correlations',
                    color='Correlation',
                    color_continuous_scale='RdYlBu_r'
                )
                fig.update_layout(template='plotly_dark', font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                if len(chem_cols) > 1:
                    corr_matrix = data[chem_cols + ['Expression_Fold_Change']].corr()
                    
                    fig_heatmap = px.imshow(
                        corr_matrix,
                        color_continuous_scale='RdYlBu_r',
                        title='Property Correlation Matrix'
                    )
                    fig_heatmap.update_layout(template='plotly_dark', font=dict(color='white'))
                    st.plotly_chart(fig_heatmap, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")
    
    def _chemical_space_analysis(self, data: pd.DataFrame, chem_cols: List[str]):
        """Chemical space analysis with PCA"""
        try:
            st.markdown("#### Chemical Space Analysis")
            
            if len(chem_cols) < 2:
                st.warning("Need at least 2 chemical descriptors for PCA analysis")
                return
            
            # Prepare data for PCA
            X = data[chem_cols].fillna(data[chem_cols].mean())
            
            # PCA
            pca = PCA(n_components=min(3, len(chem_cols)))
            X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
            
            # Create PCA dataframe
            pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
            pca_df['Expression_Fold_Change'] = data['Expression_Fold_Change'].values
            pca_df['Compound'] = data['Compound'].values
            
            # PCA scatter plot
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                color='Expression_Fold_Change',
                hover_data=['Compound'],
                title='Chemical Space (PCA)',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(template='plotly_dark', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Explained variance
            explained_var = pca.explained_variance_ratio_
            st.markdown("**PCA Explained Variance:**")
            for i, var in enumerate(explained_var):
                st.write(f"PC{i+1}: {var:.1%}")
        
        except Exception as e:
            st.error(f"Error in chemical space analysis: {str(e)}")

class DataExplorerTab:
    """Data Explorer Tab"""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
    
    def render(self, data: pd.DataFrame, hit_threshold: float):
        """Render data explorer"""
        if data is None or len(data) == 0:
            st.warning("No data available for exploration.")
            return
        
        try:
            st.markdown("### Data Explorer")
            
            # Data overview
            st.markdown("#### Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Wells", len(data))
            
            with col2:
                st.metric("Unique Compounds", data['Compound'].nunique())
            
            with col3:
                st.metric("Columns", len(data.columns))
            
            with col4:
                missing_pct = data.isnull().sum().sum() / data.size * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            
            # Data tabs
            data_tab1, data_tab2, data_tab3 = st.tabs([
                "Data Table", "Statistics", "Missing Data"
            ])
            
            with data_tab1:
                self._display_data_table(data)
            
            with data_tab2:
                self._display_statistics(data)
            
            with data_tab3:
                self._display_missing_data(data)
        
        except Exception as e:
            st.error(f"Data Explorer Error: {str(e)}")
            logger.error(f"Data explorer error: {str(e)}\n{traceback.format_exc()}")
    
    def _display_data_table(self, data: pd.DataFrame):
        """Display interactive data table"""
        try:
            st.markdown("#### Raw Data")
            
            # Column selection
            all_cols = data.columns.tolist()
            selected_cols = st.multiselect(
                "Select columns to display:",
                all_cols,
                default=all_cols[:10]  # Show first 10 by default
            )
            
            if selected_cols:
                # Row filtering
                n_rows = st.slider("Number of rows to display:", 10, min(1000, len(data)), 50)
                
                # Display table
                display_data = data[selected_cols].head(n_rows)
                st.dataframe(display_data, use_container_width=True)
                
                # Download option
                csv = display_data.to_csv(index=False)
                st.download_button(
                    label="Download filtered data as CSV",
                    data=csv,
                    file_name="filtered_hts_data.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error displaying data table: {str(e)}")
    
    def _display_statistics(self, data: pd.DataFrame):
        """Display statistical summary"""
        try:
            st.markdown("#### Statistical Summary")
            
            # Numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                stats_df = data[numeric_cols].describe()
                st.dataframe(stats_df.round(3), use_container_width=True)
                
                # Distribution plots
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("Select column for distribution:", numeric_cols)
                    
                    if selected_col:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram
                            fig_hist = px.histogram(
                                data, x=selected_col,
                                title=f'Distribution of {selected_col}',
                                nbins=30
                            )
                            fig_hist.update_layout(template='plotly_dark', font=dict(color='white'))
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with col2:
                            # Box plot
                            fig_box = px.box(
                                data, y=selected_col,
                                title=f'Box Plot of {selected_col}'
                            )
                            fig_box.update_layout(template='plotly_dark', font=dict(color='white'))
                            st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No numeric columns found for statistical analysis")
        
        except Exception as e:
            st.error(f"Error in statistical summary: {str(e)}")
    
    def _display_missing_data(self, data: pd.DataFrame):
        """Display missing data analysis"""
        try:
            st.markdown("#### Missing Data Analysis")
            
            # Missing data summary
            missing_data = data.isnull().sum()
            missing_pct = (missing_data / len(data)) * 100
            
            missing_summary = pd.DataFrame({
                'Column': missing_data.index,
                'Missing_Count': missing_data.values,
                'Missing_Percentage': missing_pct.values
            }).sort_values('Missing_Percentage', ascending=False)
            
            # Filter columns with missing data
            missing_cols = missing_summary[missing_summary['Missing_Count'] > 0]
            
            if len(missing_cols) > 0:
                st.dataframe(missing_cols.round(2), use_container_width=True, hide_index=True)
                
                # Missing data heatmap
                if len(missing_cols) <= 20:  # Only show heatmap for manageable number of columns
                    missing_matrix = data[missing_cols['Column']].isnull()
                    
                    fig_missing = px.imshow(
                        missing_matrix.T.astype(int),
                        title='Missing Data Pattern',
                        color_continuous_scale='Reds',
                        labels={'color': 'Missing'}
                    )
                    fig_missing.update_layout(template='plotly_dark', font=dict(color='white'))
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.info("Too many columns with missing data to display heatmap")
            else:
                st.success("No missing data found!")
        
        except Exception as e:
            st.error(f"Error in missing data analysis: {str(e)}")

class ReportingTab:
    """Reporting and Export Tab"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#00d4aa',
            'secondary': '#ffab00',
            'danger': '#ff5252',
            'warning': '#ff9500',
            'info': '#2196f3',
            'success': '#4caf50'
        }
    
    def render(self, data: pd.DataFrame, metrics: Dict, hit_threshold: float):
        """Render reporting tab"""
        if data is None or len(data) == 0:
            st.warning("No data available for reporting.")
            return
        
        try:
            st.markdown("### Reporting & Export")
            
            # Report generation
            report_tab1, report_tab2, report_tab3 = st.tabs([
                "Executive Summary", "Detailed Report", "Data Export"
            ])
            
            with report_tab1:
                self._generate_executive_summary(data, metrics, hit_threshold)
            
            with report_tab2:
                self._generate_detailed_report(data, metrics, hit_threshold)
            
            with report_tab3:
                self._data_export_options(data, hit_threshold)
        
        except Exception as e:
            st.error(f"Reporting Error: {str(e)}")
            logger.error(f"Reporting error: {str(e)}\n{traceback.format_exc()}")
    
    def _generate_executive_summary(self, data: pd.DataFrame, metrics: Dict, hit_threshold: float):
        """Generate executive summary"""
        try:
            st.markdown("#### Executive Summary")
            
            # Key findings
            hits = data[data['Expression_Fold_Change'] >= hit_threshold]
            
            summary_text = f"""
            ### HTS Screening Campaign Summary
            
            **Campaign Overview:**
            - **Total Wells Screened:** {len(data):,}
            - **Unique Compounds:** {data['Compound'].nunique():,}
            - **Hit Threshold:** {hit_threshold}x fold change
            
            **Key Results:**
            - **Total Hits Identified:** {len(hits):,}
            - **Hit Rate:** {len(hits)/len(data)*100:.1f}%
            - **Assay Quality (Z'-factor):** {metrics.get('z_prime', 0):.3f}
            
            **Quality Assessment:**
            """
            
            # Quality assessment
            z_prime = metrics.get('z_prime', 0)
            if z_prime >= 0.5:
                summary_text += "- **Assay Quality:** Excellent ✅\n"
            elif z_prime > 0:
                summary_text += "- **Assay Quality:** Acceptable ⚠️\n"
            else:
                summary_text += "- **Assay Quality:** Poor ❌\n"
            
            # Signal quality
            sb_ratio = metrics.get('sb_ratio', 1)
            if sb_ratio >= 2:
                summary_text += "- **Signal Quality:** Strong ✅\n"
            elif sb_ratio >= 1.5:
                summary_text += "- **Signal Quality:** Moderate ⚠️\n"
            else:
                summary_text += "- **Signal Quality:** Weak ❌\n"
            
            # Hit quality
            if len(hits) > 0:
                max_activity = hits['Expression_Fold_Change'].max()
                avg_activity = hits['Expression_Fold_Change'].mean()
                summary_text += f"""
            **Hit Analysis:**
            - **Strongest Hit:** {max_activity:.1f}x fold change
            - **Average Hit Activity:** {avg_activity:.1f}x fold change
            """
                
                # Top hits
                top_hits = hits.nlargest(5, 'Expression_Fold_Change')
                summary_text += "\n**Top 5 Hits:**\n"
                for idx, (_, hit) in enumerate(top_hits.iterrows(), 1):
                    summary_text += f"{idx}. {hit['Compound']} ({hit['Well']}): {hit['Expression_Fold_Change']:.2f}x\n"
            
            st.markdown(summary_text)
            
            # Download executive summary
            st.download_button(
                label="Download Executive Summary",
                data=summary_text,
                file_name="hts_executive_summary.md",
                mime="text/markdown"
            )
        
        except Exception as e:
            st.error(f"Error generating executive summary: {str(e)}")
    
    def _generate_detailed_report(self, data: pd.DataFrame, metrics: Dict, hit_threshold: float):
        """Generate detailed technical report"""
        try:
            st.markdown("#### Detailed Technical Report")
            
            # Comprehensive analysis
            hits = data[data['Expression_Fold_Change'] >= hit_threshold]
            
            detailed_report = f"""
            # HTS Screening Campaign - Detailed Report
            
            ## 1. Experimental Design
            - **Assay Type:** Expression fold change screening
            - **Plate Format:** {len(data)} wells
            - **Hit Threshold:** {hit_threshold}x fold change
            - **Date Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            
            ## 2. Quality Control Metrics
            """
            
            # QC section
            detailed_report += f"""
            ### 2.1 Assay Performance
            - **Z'-factor:** {metrics.get('z_prime', 0):.4f}
            - **Signal/Background Ratio:** {metrics.get('sb_ratio', 1):.2f}
            - **Coefficient of Variation:** {metrics.get('cv_percent', 0):.1f}%
            - **Dynamic Range:** {metrics.get('dynamic_range', 0):.2f}
            
            ### 2.2 Control Performance
            - **Positive Control Mean:** {metrics.get('pos_controls_mean', 1):.3f} ± {metrics.get('pos_controls_std', 0):.3f}
            - **Negative Control Mean:** {metrics.get('neg_controls_mean', 1):.3f} ± {metrics.get('neg_controls_std', 0):.3f}
            
            ## 3. Hit Identification
            """
            
            # Hits section
            if len(hits) > 0:
                # Activity distribution
                activity_ranges = [
                    (hit_threshold, 2.0, "Moderate"),
                    (2.0, 3.0, "Strong"), 
                    (3.0, float('inf'), "Very Strong")
                ]
                
                detailed_report += f"""
            ### 3.1 Hit Summary
            - **Total Hits:** {len(hits):,}
            - **Hit Rate:** {len(hits)/len(data)*100:.2f}%
            - **Average Hit Activity:** {hits['Expression_Fold_Change'].mean():.2f}x
            - **Maximum Hit Activity:** {hits['Expression_Fold_Change'].max():.2f}x
            
            ### 3.2 Activity Distribution
            """
                
                for min_val, max_val, category in activity_ranges:
                    if max_val == float('inf'):
                        category_hits = hits[hits['Expression_Fold_Change'] >= min_val]
                        range_text = f">= {min_val}x"
                    else:
                        category_hits = hits[
                            (hits['Expression_Fold_Change'] >= min_val) & 
                            (hits['Expression_Fold_Change'] < max_val)
                        ]
                        range_text = f"{min_val}-{max_val}x"
                    
                    detailed_report += f"- **{category} ({range_text}):** {len(category_hits)} hits ({len(category_hits)/len(hits)*100:.1f}%)\n"
                
                # Top hits table
                top_hits = hits.nlargest(10, 'Expression_Fold_Change')
                detailed_report += f"""
            
            ### 3.3 Top 10 Hits
            | Rank | Well | Compound | Activity (x-fold) |
            |------|------|----------|------------------|
            """
                
                for idx, (_, hit) in enumerate(top_hits.iterrows(), 1):
                    detailed_report += f"| {idx} | {hit['Well']} | {hit['Compound']} | {hit['Expression_Fold_Change']:.2f} |\n"
            
            else:
                detailed_report += f"""
            ### 3.1 Hit Summary
            - **No hits identified** with the current threshold of {hit_threshold}x
            - Consider lowering the threshold or reviewing assay conditions
            """
            
            # Statistical analysis
            detailed_report += f"""
            
            ## 4. Statistical Analysis
            ### 4.1 Data Distribution
            - **Mean Expression:** {data['Expression_Fold_Change'].mean():.3f}
            - **Median Expression:** {data['Expression_Fold_Change'].median():.3f}
            - **Standard Deviation:** {data['Expression_Fold_Change'].std():.3f}
            - **Skewness:** {data['Expression_Fold_Change'].skew():.3f}
            - **Kurtosis:** {data['Expression_Fold_Change'].kurtosis():.3f}
            
            ### 4.2 Outlier Analysis
            """
            
            # Outlier analysis
            Q1 = data['Expression_Fold_Change'].quantile(0.25)
            Q3 = data['Expression_Fold_Change'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data['Expression_Fold_Change'] < lower_bound) | 
                           (data['Expression_Fold_Change'] > upper_bound)]
            
            detailed_report += f"- **Outliers (1.5×IQR rule):** {len(outliers)} compounds ({len(outliers)/len(data)*100:.1f}%)\n"
            detailed_report += f"- **IQR Range:** {Q1:.3f} - {Q3:.3f}\n"
            detailed_report += f"- **Outlier Bounds:** < {lower_bound:.3f} or > {upper_bound:.3f}\n"
            
            # Recommendations
            detailed_report += f"""
            
            ## 5. Recommendations
            """
            
            recommendations = []
            
            # Based on Z'-factor
            z_prime = metrics.get('z_prime', 0)
            if z_prime < 0:
                recommendations.append("- **Assay Optimization Required:** Z'-factor < 0 indicates poor assay performance. Consider optimizing controls, reducing variability, or increasing signal window.")
            elif z_prime < 0.5:
                recommendations.append("- **Assay Improvement Suggested:** Z'-factor between 0-0.5 indicates marginal performance. Minor optimizations may improve reliability.")
            else:
                recommendations.append("- **Assay Performance Excellent:** Z'-factor > 0.5 indicates high-quality assay suitable for screening.")
            
            # Based on hit rate
            hit_rate = len(hits) / len(data) * 100
            if hit_rate == 0:
                recommendations.append("- **No Hits Found:** Consider lowering hit threshold, expanding compound library, or reviewing assay sensitivity.")
            elif hit_rate < 1:
                recommendations.append("- **Low Hit Rate:** Consider optimizing screening conditions or expanding chemical diversity.")
            elif hit_rate > 10:
                recommendations.append("- **High Hit Rate:** May indicate assay artifacts or non-specific activity. Consider tightening hit criteria.")
            else:
                recommendations.append("- **Optimal Hit Rate:** Hit rate in acceptable range for follow-up studies.")
            
            # Follow-up recommendations
            if len(hits) > 0:
                recommendations.append("- **Follow-up Studies:** Prioritize hits for dose-response confirmation, orthogonal assays, and structure-activity relationship analysis.")
                recommendations.append("- **Hit Validation:** Confirm hit activity with fresh compound stocks and independent assays.")
            
            for rec in recommendations:
                detailed_report += rec + "\n"
            
            st.markdown(detailed_report)
            
            # Download detailed report
            st.download_button(
                label="Download Detailed Report",
                data=detailed_report,
                file_name="hts_detailed_report.md",
                mime="text/markdown"
            )
        
        except Exception as e:
            st.error(f"Error generating detailed report: {str(e)}")
    
    def _data_export_options(self, data: pd.DataFrame, hit_threshold: float):
        """Provide data export options"""
        try:
            st.markdown("#### Data Export Options")
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Hit Data Export**")
                
                hits = data[data['Expression_Fold_Change'] >= hit_threshold]
                
                if len(hits) > 0:
                    # Hits CSV
                    hits_csv = hits.to_csv(index=False)
                    st.download_button(
                        label=f"Download Hits Data ({len(hits)} compounds)",
                        data=hits_csv,
                        file_name="hts_hits.csv",
                        mime="text/csv"
                    )
                    
                    # Top hits only
                    top_hits = hits.nlargest(20, 'Expression_Fold_Change')
                    top_hits_csv = top_hits.to_csv(index=False)
                    st.download_button(
                        label="Download Top 20 Hits",
                        data=top_hits_csv,
                        file_name="hts_top_hits.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No hits to export")
            
            with col2:
                st.markdown("**📊 Complete Dataset Export**")
                
                # Full dataset
                full_csv = data.to_csv(index=False)
                st.download_button(
                    label=f"Download Complete Dataset ({len(data)} wells)",
                    data=full_csv,
                    file_name="hts_complete_data.csv",
                    mime="text/csv"
                )
                
                # Filtered by compound type
                test_compounds = data[~data['Compound'].str.contains('Control|DMSO', case=False, na=False)]
                if len(test_compounds) > 0:
                    test_csv = test_compounds.to_csv(index=False)
                    st.download_button(
                        label=f"Download Test Compounds Only ({len(test_compounds)} wells)",
                        data=test_csv,
                        file_name="hts_test_compounds.csv",
                        mime="text/csv"
                    )
            
            # Export formats
            st.markdown("#### Advanced Export Options")
            
            export_format = st.selectbox(
                "Select Export Format:",
                ["CSV", "Excel", "JSON", "Parquet"]
            )
            
            include_metadata = st.checkbox("Include Analysis Metadata", value=True)
            
            if st.button("Generate Custom Export", type="primary"):
                try:
                    if export_format == "CSV":
                        export_data = data.to_csv(index=False)
                        mime_type = "text/csv"
                        file_ext = ".csv"
                    elif export_format == "Excel":
                        from io import BytesIO
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            data.to_excel(writer, sheet_name='HTS_Data', index=False)
                            if len(data[data['Expression_Fold_Change'] >= hit_threshold]) > 0:
                                hits.to_excel(writer, sheet_name='Hits', index=False)
                        export_data = buffer.getvalue()
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        file_ext = ".xlsx"
                    elif export_format == "JSON":
                        export_data = data.to_json(orient='records', indent=2)
                        mime_type = "application/json"
                        file_ext = ".json"
                    elif export_format == "Parquet":
                        from io import BytesIO
                        buffer = BytesIO()
                        data.to_parquet(buffer, index=False)
                        export_data = buffer.getvalue()
                        mime_type = "application/octet-stream"
                        file_ext = ".parquet"
                    
                    st.download_button(
                        label=f"Download {export_format} File",
                        data=export_data,
                        file_name=f"hts_export{file_ext}",
                        mime=mime_type
                    )
                    
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Error in data export: {str(e)}")

# Additional utility functions
def dose_response_curve(concentrations, ic50, hill_slope, top, bottom):
    """4-parameter logistic dose-response curve"""
    return bottom + (top - bottom) / (1 + (concentrations / ic50) ** hill_slope)

def fit_dose_response(concentrations, responses):
    """Fit dose-response curve to data"""
    try:
        # Initial parameter estimates
        top_init = np.max(responses)
        bottom_init = np.min(responses)
        ic50_init = np.median(concentrations)
        hill_init = 1.0
        
        # Fit curve
        popt, _ = curve_fit(
            dose_response_curve,
            concentrations, responses,
            p0=[ic50_init, hill_init, top_init, bottom_init],
            maxfev=5000
        )
        
        return popt
    except:
        return None