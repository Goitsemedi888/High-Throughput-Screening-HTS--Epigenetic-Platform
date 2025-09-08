import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings
import logging
import traceback
from typing import Dict, List, Optional, Tuple, Union

# Add these imports to hts_dashboard_fixed.py
from enhanced_ml_pipeline import EnhancedMLPipeline, MLConfig
from enhanced_visualization import ProductionHTSVisualizer, EnhancedMLVisualizer
from hts_remaining_tabs import (
    HitIdentificationTab, MLAnalysisTab, SARAnalysisTab, 
    DataExplorerTab, ReportingTab
)
from enhanced_ml_tab import EnhancedMLAnalysisTab

from production_utils import (
    AdvancedDataValidator, ValidationResult,
    retry_on_failure, log_execution_time, error_handler,
    CacheManager, cached,
    PerformanceMonitor, AdvancedErrorHandler,
    BenchmarkSuite
)
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="HTS Epigenetic Regulator Discovery Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary-color: #00d4aa;
        --secondary-color: #0066cc;
        --accent-color: #ff6b6b;
        --bg-primary: #0e1117;
        --bg-secondary: #1a1d29;
        --bg-tertiary: #262730;
        --text-primary: #ffffff;
        --text-secondary: #a0a4b8;
        --border-color: #333645;
        --glass-bg: rgba(26, 29, 41, 0.85);
        --glass-border: rgba(255, 255, 255, 0.1);
        --error-color: #ff5252;
        --warning-color: #ffab00;
        --success-color: #00d4aa;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 212, 170, 0.3);
    }
    
    .sub-header {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .metric-card, .custom-metric {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover, .custom-metric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 212, 170, 0.2);
        border-color: var(--primary-color);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .error-box {
        background: rgba(255, 82, 82, 0.1);
        border: 1px solid var(--error-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(255, 171, 0, 0.1);
        border: 1px solid var(--warning-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(0, 212, 170, 0.1);
        border: 1px solid var(--success-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-tertiary);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
        padding: 12px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        box-shadow: 0 4px 16px rgba(0, 212, 170, 0.3);
    }
    
    [data-testid="metric-container"] {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 212, 170, 0.15);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 212, 170, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 212, 170, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 class="main-header">🧬 HTS Epigenetic Regulator Discovery Platform</h1>
    <p class="sub-header">
        <strong>Production-Ready Computational Biology & Advanced Machine Learning</strong>
    </p>
    <div style="height: 3px; background: linear-gradient(90deg, transparent, #00d4aa, #0066cc, transparent); margin: 2rem auto; width: 60%; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)

class HTSAnalyzer:
    """
    High-Throughput Screening Analysis Class
    
    Provides comprehensive analysis capabilities for HTS data including:
    - Data validation and preprocessing
    - Statistical analysis and quality control
    - Machine learning model training and evaluation
    - Structure-Activity Relationship analysis
    - Dose-response curve fitting
    """
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate uploaded HTS data format and content
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        required_columns = ['Well', 'Compound', 'Expression_Fold_Change']
        
        try:
            # Check required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                errors.append(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Check data types and ranges
            if 'Expression_Fold_Change' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['Expression_Fold_Change']):
                    errors.append("Expression_Fold_Change must be numeric")
                elif df['Expression_Fold_Change'].isna().all():
                    errors.append("Expression_Fold_Change contains no valid values")
                # Note: Removed negative value check as fold changes can be < 1 for inhibitors
            
            # Check for duplicates
            if 'Well' in df.columns and df['Well'].duplicated().any():
                errors.append("Duplicate wells found in data")
            
            # Validate well format if present
            if 'Well' in df.columns:
                invalid_wells = df[~df['Well'].str.match(r'^[A-P]\d{2}$', na=False)]
                if len(invalid_wells) > 0:
                    errors.append(f"Invalid well format found. Expected format: A01-P24")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            errors.append(f"Data validation error: {str(e)}")
            return False, errors
    
    def generate_chemical_descriptors(self, compound_names: List[str]) -> pd.DataFrame:
        """
        Generate realistic chemical descriptors for compounds
        
        Args:
            compound_names: List of compound names
            
        Returns:
            DataFrame with chemical descriptors
        """
        try:
            np.random.seed(42)
            n_compounds = len(compound_names)
            
            # Generate correlated descriptors for more realism
            base_complexity = np.random.normal(0.5, 0.2, n_compounds)
            
            descriptors = {
                'Compound': compound_names,
                'MW': np.random.normal(350, 100, n_compounds),
                'LogP': np.random.normal(2.5, 1.5, n_compounds),
                'HBD': np.random.poisson(2, n_compounds),
                'HBA': np.random.poisson(4, n_compounds),
                'TPSA': np.random.normal(75, 30, n_compounds),
                'RotBonds': np.random.poisson(5, n_compounds),
                'AromaticRings': np.random.poisson(2, n_compounds),
                'Complexity': base_complexity,
                # Advanced descriptors
                'SlogP': np.random.normal(2.0, 1.2, n_compounds),
                'NumHeavyAtoms': np.random.normal(25, 5, n_compounds),
                'FractionCsp3': np.random.beta(2, 2, n_compounds),
                'NumAliphaticRings': np.random.poisson(1, n_compounds)
            }
            
            # Ensure realistic constraints
            df = pd.DataFrame(descriptors)
            df['MW'] = np.clip(df['MW'], 150, 800)
            df['LogP'] = np.clip(df['LogP'], -3, 8)
            df['TPSA'] = np.clip(df['TPSA'], 0, 200)
            df['NumHeavyAtoms'] = np.clip(df['NumHeavyAtoms'], 10, 50).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating chemical descriptors: {str(e)}")
            raise

@st.cache_data
def generate_hts_data(seed: int = 42, n_compounds: int = 20) -> pd.DataFrame:
    """
    Generate realistic HTS screening data with proper biological variation
    
    Args:
        seed: Random seed for reproducibility
        n_compounds: Number of test compounds
        
    Returns:
        DataFrame with HTS screening data
    """
    try:
        np.random.seed(seed)
        
        # Create 384-well plate layout
        rows = list("ABCDEFGHIJKLMNOP")
        wells = [f"{row}{col:02d}" for row in rows for col in range(1, 25)]
        
        # Define well assignments with realistic biological controls
        n_wells = 384
        n_pos_controls = 32  # Strong activators (should give ~3-5x fold change)
        n_neg_controls = 32  # Vehicle controls (should be ~1x fold change)
        n_dmso = 16         # DMSO controls (should be ~1x fold change)
        n_test_wells = n_wells - n_pos_controls - n_neg_controls - n_dmso
        
        # Create compound assignments
        compound_types = []
        
        # Positive controls - should show strong activation
        compound_types.extend(["Control+"] * n_pos_controls)
        
        # Negative controls - should show baseline activity
        compound_types.extend(["Control-"] * n_neg_controls) 
        
        # DMSO controls - should show baseline activity
        compound_types.extend(["DMSO"] * n_dmso)
        
        # Test compounds
        wells_per_compound = max(1, n_test_wells // n_compounds)
        remaining_wells = n_test_wells % n_compounds
        
        for i in range(n_compounds):
            n_wells_this_compound = wells_per_compound + (1 if i < remaining_wells else 0)
            compound_types.extend([f"Compound_{i+1:03d}"] * n_wells_this_compound)
        
        # Pad if needed
        while len(compound_types) < n_wells:
            compound_types.append(f"Compound_{len([c for c in compound_types if c.startswith('Compound_')]) + 1:03d}")
        
        compound_types = compound_types[:n_wells]
        
        # Shuffle for randomized plate layout
        combined = list(zip(wells, compound_types))
        np.random.shuffle(combined)
        wells, compound_types = zip(*combined)
        
        # Generate realistic expression data with proper biological signals
        data = pd.DataFrame({
            "Well": list(wells),
            "Row": [well[0] for well in wells],
            "Column": [int(well[1:]) for well in wells],
            "Compound": list(compound_types),
            "Plate_ID": ["Plate_001"] * len(wells),
            "Assay_Date": ["2024-01-15"] * len(wells)
        })
        
        # Initialize expression with realistic baseline noise
        base_expression = np.random.normal(1.0, 0.15, len(data))
        
        # Add systematic plate effects (edge effects, gradients)
        for i, well in enumerate(data['Well']):
            row_idx = ord(well[0]) - ord('A')
            col_idx = int(well[1:]) - 1
            
            # Edge effect (wells on edges tend to have slightly different values)
            if row_idx == 0 or row_idx == 15 or col_idx == 0 or col_idx == 23:
                base_expression[i] *= np.random.normal(1.05, 0.08)
            
            # Subtle plate gradient
            gradient_effect = 1.0 + 0.02 * (row_idx / 15.0) + 0.01 * (col_idx / 23.0)
            base_expression[i] *= gradient_effect
        
        data['Expression_Fold_Change'] = base_expression
        
        # Apply realistic control effects with proper separation
        # Positive controls: Strong activators (3-5x fold change)
        pos_mask = data["Compound"] == "Control+"
        if pos_mask.any():
            pos_effects = np.random.normal(4.0, 0.8, pos_mask.sum())
            pos_effects = np.clip(pos_effects, 2.5, 6.0)  # Ensure reasonable range
            data.loc[pos_mask, "Expression_Fold_Change"] = pos_effects
        
        # Negative controls: Baseline activity (0.8-1.2x fold change)
        neg_mask = data["Compound"] == "Control-"
        if neg_mask.any():
            neg_effects = np.random.normal(1.0, 0.15, neg_mask.sum())
            neg_effects = np.clip(neg_effects, 0.7, 1.3)
            data.loc[neg_mask, "Expression_Fold_Change"] = neg_effects
        
        # DMSO controls: Similar to negative controls
        dmso_mask = data["Compound"] == "DMSO"
        if dmso_mask.any():
            dmso_effects = np.random.normal(1.0, 0.12, dmso_mask.sum())
            dmso_effects = np.clip(dmso_effects, 0.8, 1.2)
            data.loc[dmso_mask, "Expression_Fold_Change"] = dmso_effects
        
        # Test compounds: Mix of inactive and active compounds
        test_compounds = [c for c in data["Compound"].unique() if c.startswith("Compound_")]
        
        for compound in test_compounds:
            comp_mask = data["Compound"] == compound
            n_wells_comp = comp_mask.sum()
            
            # 20% chance of being a hit compound
            if np.random.random() < 0.20:
                # Hit compound: 1.5-4x fold change
                activity_level = np.random.choice([1.8, 2.5, 3.2, 4.0], p=[0.4, 0.3, 0.2, 0.1])
                comp_effects = np.random.normal(activity_level, 0.3, n_wells_comp)
                comp_effects = np.clip(comp_effects, 1.2, 5.0)
                data.loc[comp_mask, "Expression_Fold_Change"] = comp_effects
            else:
                # Inactive compound: 0.7-1.3x fold change
                comp_effects = np.random.normal(1.0, 0.2, n_wells_comp)
                comp_effects = np.clip(comp_effects, 0.5, 1.4)
                data.loc[comp_mask, "Expression_Fold_Change"] = comp_effects
        
        # Add quality control metrics
        data["Z_Score"] = stats.zscore(data["Expression_Fold_Change"])
        
        # Calculate CV properly
        mean_expr = data["Expression_Fold_Change"].mean()
        std_expr = data["Expression_Fold_Change"].std()
        data["CV"] = std_expr / mean_expr if mean_expr != 0 else 0
        
        # Add chemical descriptors for test compounds
        if test_compounds:
            analyzer = HTSAnalyzer()
            chem_descriptors = analyzer.generate_chemical_descriptors(test_compounds)
            data = data.merge(chem_descriptors, on='Compound', how='left')
        
        # Add concentration information for dose-response analysis
        data['Concentration_uM'] = 10.0  # Default screening concentration
        
        # Ensure we have some variability in the data
        if data['Expression_Fold_Change'].std() < 0.1:
            # Add some controlled variability if data is too uniform
            noise = np.random.normal(0, 0.2, len(data))
            data['Expression_Fold_Change'] += noise
            data['Expression_Fold_Change'] = np.clip(data['Expression_Fold_Change'], 0.1, 10.0)
            
            # Recalculate Z-scores
            data["Z_Score"] = stats.zscore(data["Expression_Fold_Change"])
        
        logger.info(f"Generated HTS data with {len(data)} wells")
        logger.info(f"Expression range: {data['Expression_Fold_Change'].min():.2f} - {data['Expression_Fold_Change'].max():.2f}")
        logger.info(f"Expression mean ± std: {data['Expression_Fold_Change'].mean():.2f} ± {data['Expression_Fold_Change'].std():.2f}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error generating HTS data: {str(e)}")
        st.error(f"Error generating demo data: {str(e)}")
        return pd.DataFrame()

def load_csv_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Load and validate CSV data from uploaded file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (DataFrame or None, list of error messages)
    """
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Initialize analyzer and validate
        analyzer = HTSAnalyzer()
        is_valid, errors = analyzer.validate_data(df)
        
        if not is_valid:
            return None, errors
        
        # Add missing columns with defaults
        if 'Row' not in df.columns and 'Well' in df.columns:
            df['Row'] = df['Well'].str[0]
        
        if 'Column' not in df.columns and 'Well' in df.columns:
            df['Column'] = df['Well'].str[1:].astype(int)
        
        if 'Z_Score' not in df.columns and 'Expression_Fold_Change' in df.columns:
            df['Z_Score'] = stats.zscore(df['Expression_Fold_Change'])
        
        if 'CV' not in df.columns and 'Expression_Fold_Change' in df.columns:
            mean_expr = df['Expression_Fold_Change'].mean()
            std_expr = df['Expression_Fold_Change'].std()
            df['CV'] = std_expr / mean_expr if mean_expr != 0 else 0
        
        # Generate chemical descriptors if not present
        test_compounds = [c for c in df['Compound'].unique() if not c.lower().startswith(('control', 'dmso', 'blank'))]
        
        if test_compounds and 'MW' not in df.columns:
            try:
                chem_descriptors = analyzer.generate_chemical_descriptors(test_compounds)
                df = df.merge(chem_descriptors, on='Compound', how='left')
            except Exception as e:
                logger.warning(f"Could not generate chemical descriptors: {str(e)}")
        
        logger.info(f"Successfully loaded CSV data with {len(df)} rows and {len(df.columns)} columns")
        return df, []
        
    except Exception as e:
        error_msg = f"Error loading CSV file: {str(e)}"
        logger.error(error_msg)
        return None, [error_msg]

@st.cache_data
def calculate_metrics(df: pd.DataFrame, hit_threshold: float) -> Dict:
    """
    Calculate comprehensive screening metrics with error handling
    
    Args:
        df: HTS data DataFrame
        hit_threshold: Fold change threshold for hit identification
        
    Returns:
        Dictionary of calculated metrics
    """
    try:
        # Get control data - handle both exact matches and partial matches
        pos_controls = df[df["Compound"].str.contains("Control\\+", regex=True, na=False)]["Expression_Fold_Change"]
        neg_controls = df[df["Compound"].str.contains("Control-", regex=True, na=False)]["Expression_Fold_Change"]
        
        # Fallback: look for any positive/negative control indicators
        if len(pos_controls) == 0:
            pos_controls = df[df["Compound"].str.contains("pos", case=False, na=False)]["Expression_Fold_Change"]
        if len(neg_controls) == 0:
            neg_controls = df[df["Compound"].str.contains("neg|dmso", case=False, na=False)]["Expression_Fold_Change"]
        
        # Calculate control statistics
        if len(pos_controls) == 0 or len(neg_controls) == 0:
            st.warning("⚠️ Missing or insufficient control data. Using data quartiles for Z'-factor calculation.")
            # Use quartiles as proxy for controls
            pos_mean, pos_std = df["Expression_Fold_Change"].quantile(0.75), df["Expression_Fold_Change"].std() * 0.5
            neg_mean, neg_std = df["Expression_Fold_Change"].quantile(0.25), df["Expression_Fold_Change"].std() * 0.5
        else:
            pos_mean, pos_std = pos_controls.mean(), pos_controls.std()
            neg_mean, neg_std = neg_controls.mean(), neg_controls.std()
        
        # Ensure we have valid standard deviations
        pos_std = max(pos_std, 0.001) if not np.isnan(pos_std) else 0.1
        neg_std = max(neg_std, 0.001) if not np.isnan(neg_std) else 0.1
        
        # Z'-factor calculation with error handling
        try:
            denominator = abs(pos_mean - neg_mean)
            if denominator > 0:
                z_prime = 1 - (3 * (pos_std + neg_std)) / denominator
            else:
                z_prime = 0
                logger.warning("Zero denominator in Z'-factor calculation")
        except (ZeroDivisionError, TypeError):
            z_prime = 0
            logger.warning("Could not calculate Z'-factor")
        
        # Hit rate calculation
        hits = df[df["Expression_Fold_Change"] >= hit_threshold]
        hit_rate = len(hits) / len(df) * 100 if len(df) > 0 else 0
        
        # Signal-to-background ratio
        sb_ratio = pos_mean / neg_mean if neg_mean > 0 else pos_mean
        
        # Additional quality metrics
        mean_expr = df["Expression_Fold_Change"].mean()
        std_expr = df["Expression_Fold_Change"].std()
        coefficient_of_variation = (std_expr / mean_expr) * 100 if mean_expr > 0 else 0
        dynamic_range = df["Expression_Fold_Change"].max() - df["Expression_Fold_Change"].min()
        
        # Median Absolute Deviation
        median_expr = df["Expression_Fold_Change"].median()
        mad = df["Expression_Fold_Change"].sub(median_expr).abs().median()
        
        return {
            "z_prime": z_prime,
            "hit_rate": hit_rate,
            "sb_ratio": sb_ratio,
            "total_wells": len(df),
            "pos_controls_mean": pos_mean,
            "neg_controls_mean": neg_mean,
            "pos_controls_std": pos_std,
            "neg_controls_std": neg_std,
            "cv_percent": coefficient_of_variation,
            "dynamic_range": dynamic_range,
            "n_hits": len(hits),
            "median_expression": median_expr,
            "mad": mad,
            "mean_expression": mean_expr,
            "std_expression": std_expr
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        # Return safe defaults
        return {
            "z_prime": 0, "hit_rate": 0, "sb_ratio": 1, "total_wells": len(df),
            "pos_controls_mean": 1, "neg_controls_mean": 1, "pos_controls_std": 0.1,
            "neg_controls_std": 0.1, "cv_percent": 0, "dynamic_range": 0,
            "n_hits": 0, "median_expression": 1, "mad": 0, "mean_expression": 1, "std_expression": 0
        }

# Sidebar configuration
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; margin-bottom: 1rem; 
            background: linear-gradient(135deg, #00d4aa, #0066cc); 
            border-radius: 12px; color: white;">
    <h3>📬 Data Source & Configuration</h3>
</div>
""", unsafe_allow_html=True)

# Data source selection
data_source = st.sidebar.radio(
    "Select Data Source:",
    ["📊 Simulated Data", "📁 Upload CSV File"],
    help="Choose between generated demo data or upload your own CSV file"
)

if data_source == "📁 Upload CSV File":
    st.sidebar.markdown("### 📁 File Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload HTS screening data in CSV format"
    )
    
    if uploaded_file is not None:
        with st.sidebar.expander("📋 CSV Format Requirements"):
            st.markdown("""
            **Required columns:**
            - `Well`: Well position (e.g., A01, B12)  
            - `Compound`: Compound identifier
            - `Expression_Fold_Change`: Numeric activity data
            
            **Optional columns:**
            - `Row`, `Column`: Plate coordinates
            - Chemical descriptors (MW, LogP, etc.)
            - `Concentration_uM`: For dose-response
            - `Plate_ID`, `Assay_Date`: Metadata
            """)
        
        # Load and validate CSV data
        try:
            data, errors = load_csv_data(uploaded_file)
            
            if errors:
                for error in errors:
                    st.sidebar.error(f"❌ {error}")
                data = None
            else:
                st.sidebar.success(f"✅ Successfully loaded {len(data)} wells")
                
        except Exception as e:
            st.sidebar.error(f"❌ File loading error: {str(e)}")
            data = None
    else:
        data = None
        st.sidebar.info("👆 Please upload a CSV file to begin analysis")

else:
    # Simulated data parameters
    st.sidebar.markdown("### 🎯 Simulation Parameters")
    seed = st.sidebar.number_input("🎲 Random Seed", min_value=1, max_value=1000, value=42, step=1)
    n_compounds = st.sidebar.slider("🧪 Number of Test Compounds", min_value=5, max_value=50, value=20)
    
    # Generate simulated data
    data = generate_hts_data(seed=seed, n_compounds=n_compounds)

# After data validation and metrics calculation
if data is not None and len(data) > 0:
    # Analysis parameters (common to both data sources)
    st.sidebar.markdown("### ⚙️ Analysis Parameters")
    hit_threshold = st.sidebar.slider(
        "⚡ Hit Threshold (Fold Change)", 
        min_value=1.1, max_value=5.0, value=1.5, step=0.1,
        help="Minimum fold change to classify a compound as a hit"
    )
    
    confidence_level = st.sidebar.selectbox(
        "📊 Confidence Level", 
        [0.90, 0.95, 0.99], 
        index=1,
        format_func=lambda x: f"{x*100:.0f}%"
    )
    
    st.sidebar.markdown("---")
    
    # Advanced ML options - DEFINE THESE FIRST
    with st.sidebar.expander("🤖 Advanced ML Options"):
        enable_hyperopt = st.checkbox("Enable Hyperparameter Tuning", value=True)
        enable_shap = st.checkbox("Enable SHAP Explanations", value=True)
        enable_cv = st.checkbox("Enable Cross-Validation", value=True)
    
    # NOW initialize MLConfig with the defined variables
    ml_config = MLConfig(
        enable_hyperparameter_tuning=enable_hyperopt,
        enable_shap=enable_shap,
        enable_feature_selection=True,
        cv_folds=5 if enable_cv else 3
    )
    
    # Core pipeline and visualizers
    ml_pipeline = EnhancedMLPipeline(ml_config)
    hts_visualizer = ProductionHTSVisualizer()
    ml_visualizer = EnhancedMLVisualizer()
    
    # Initialize tab handlers
    hit_tab = HitIdentificationTab(hts_visualizer)
    ml_tab = MLAnalysisTab(ml_pipeline, ml_visualizer)
    enhanced_ml_tab = EnhancedMLAnalysisTab()
    sar_tab = SARAnalysisTab(hts_visualizer)
    explorer_tab = DataExplorerTab(hts_visualizer)
    report_tab = ReportingTab()
    
    st.sidebar.markdown("""
    <div style="padding: 1rem; background: rgba(0, 212, 170, 0.1); border-radius: 8px; border-left: 4px solid #00d4aa;">
        <h4 style="color: #00d4aa; margin: 0;">🚀 Features</h4>
        <ul style="font-size: 0.9rem; margin: 0.5rem 0;">
            <li>Advanced ML with SHAP explanations</li>
            <li>Hyperparameter optimization</li>
            <li>Model comparison & validation</li>
            <li>Robust error handling</li>
            <li>Production-ready pipeline</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate metrics
    try:
        metrics = calculate_metrics(data, hit_threshold)
        
        # Main metrics display
        st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            quality = 'Excellent' if metrics['z_prime'] > 0.5 else 'Good' if metrics['z_prime'] > 0 else 'Poor'
            quality_color = '#00d4aa' if metrics['z_prime'] > 0.5 else '#ffab00' if metrics['z_prime'] > 0 else '#ff5252'
            st.markdown(f"""
            <div class="custom-metric">
                <div class="metric-value">{metrics['z_prime']:.3f}</div>
                <div class="metric-label">Z'-Factor</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; color: {quality_color};">{quality}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="custom-metric">
                <div class="metric-value">{metrics['hit_rate']:.1f}%</div>
                <div class="metric-label">Hit Rate</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; color: #a0a4b8;">{metrics['n_hits']} hits</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            signal_quality = 'Strong' if metrics['sb_ratio'] > 2 else 'Moderate' if metrics['sb_ratio'] > 1.5 else 'Weak'
            signal_color = '#00d4aa' if metrics['sb_ratio'] > 2 else '#ffab00' if metrics['sb_ratio'] > 1.5 else '#ff5252'
            st.markdown(f"""
            <div class="custom-metric">
                <div class="metric-value">{metrics['sb_ratio']:.2f}</div>
                <div class="metric-label">Signal/Background</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; color: {signal_color};">{signal_quality}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            cv_quality = 'Excellent' if metrics['cv_percent'] < 10 else 'Good' if metrics['cv_percent'] < 20 else 'Poor'
            cv_color = '#00d4aa' if metrics['cv_percent'] < 10 else '#ffab00' if metrics['cv_percent'] < 20 else '#ff5252'
            st.markdown(f"""
            <div class="custom-metric">
                <div class="metric-value">{metrics['cv_percent']:.1f}%</div>
                <div class="metric-label">CV</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; color: {cv_color};">{cv_quality}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="custom-metric">
                <div class="metric-value">{metrics['total_wells']}</div>
                <div class="metric-label">Total Wells</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; color: #a0a4b8;">
                    {len(data['Plate_ID'].unique()) if 'Plate_ID' in data.columns else 1} plate(s)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        logger.error(f"Metrics calculation error: {str(e)}")

    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Plate Analysis", "📈 Quality Control", "🎯 Hit Identification", 
        "🤖 Advanced ML", "💊 SAR Analysis", "📋 Data Explorer", "📊 Reporting"
    ])

    with tab1:  # Plate Analysis
        st.markdown("### Plate Analysis & Heatmaps")
        hts_visualizer.render_plate_analysis(data, hit_threshold)

    with tab2:  # Quality Control  
        st.markdown("### Quality Control Metrics")
        hts_visualizer.render_quality_control(data, metrics, confidence_level)

    with tab3:  # Hit Identification
        hit_tab.render(data, hit_threshold)

    with tab4:  # Advanced ML
         if st.toggle("Use Enhanced ML Pipeline", value=True):
            enhanced_ml_tab.render(data, hit_threshold)
         else:
            ml_tab.render(data, hit_threshold, enable_shap)

    with tab5:  # SAR Analysis
        sar_tab.render(data, hit_threshold)

    with tab6:  # Data Explorer
        explorer_tab.render(data, hit_threshold)

    with tab7:  # Reporting
        report_tab.render(data, metrics, hit_threshold)

else:
    st.info("📄 Please configure data source in the sidebar to begin analysis")
    st.stop()
