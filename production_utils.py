#=============================================================================
# production_utils.py
"""
Production-ready utilities with comprehensive error handling, logging, and testing
"""

import pandas as pd
import numpy as np
import logging
import functools
import time
import traceback
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from contextlib import contextmanager
import sqlite3
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hts_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class HTSException(Exception):
    """Custom exception for HTS-specific errors"""
    pass

class DataValidationError(HTSException):
    """Exception for data validation errors"""
    pass

class ModelTrainingError(HTSException):
    """Exception for model training errors"""
    pass

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    exceptions: Tuple = (Exception,)):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                        
            raise last_exception
        return wrapper
    return decorator

def log_execution_time(func):
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

@contextmanager
def error_handler(operation_name: str, raise_on_error: bool = True):
    """Context manager for consistent error handling"""
    try:
        logger.info(f"Starting {operation_name}")
        yield
        logger.info(f"Completed {operation_name} successfully")
    except Exception as e:
        logger.error(f"Error in {operation_name}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        if raise_on_error:
            raise HTSException(f"Operation '{operation_name}' failed: {str(e)}") from e
        else:
            logger.warning(f"Continuing despite error in {operation_name}")

class AdvancedDataValidator:
    """Comprehensive data validation for HTS datasets"""
    
    def __init__(self):
        self.required_columns = ['Well', 'Compound', 'Expression_Fold_Change']
        self.optional_columns = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'AromaticRings']
        self.control_types = ['Control+', 'Control-', 'DMSO']
    
    def validate_dataset(self, data: pd.DataFrame) -> ValidationResult:
        """Comprehensive dataset validation"""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            with error_handler("Dataset Validation", raise_on_error=False):
                # Basic structure validation
                structure_errors = self._validate_structure(data)
                errors.extend(structure_errors)
                
                # Data quality validation
                quality_errors, quality_warnings = self._validate_data_quality(data)
                errors.extend(quality_errors)
                warnings.extend(quality_warnings)
                
                # Chemical descriptor validation
                if self._has_chemical_data(data):
                    chem_errors, chem_warnings = self._validate_chemical_descriptors(data)
                    errors.extend(chem_errors)
                    warnings.extend(chem_warnings)
                
                # Control validation
                control_errors, control_warnings = self._validate_controls(data)
                errors.extend(control_errors)
                warnings.extend(control_warnings)
                
                # Statistical validation
                stat_warnings = self._validate_statistics(data)
                warnings.extend(stat_warnings)
                
                # Generate metadata
                metadata = self._generate_metadata(data)
                
        except Exception as e:
            errors.append(f"Validation process failed: {str(e)}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def _validate_structure(self, data: pd.DataFrame) -> List[str]:
        """Validate basic data structure"""
        errors = []
        
        if data.empty:
            errors.append("Dataset is empty")
            return errors
        
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if 'Expression_Fold_Change' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['Expression_Fold_Change']):
                errors.append("Expression_Fold_Change must be numeric")
        
        # Check for completely empty columns
        empty_cols = data.columns[data.isnull().all()].tolist()
        if empty_cols:
            errors.append(f"Completely empty columns found: {empty_cols}")
        
        return errors
    
    def _validate_data_quality(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate data quality"""
        errors = []
        warnings = []
        
        # Check for excessive missing values
        missing_pct = data.isnull().sum() / len(data) * 100
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            warnings.append(f"Columns with >50% missing values: {high_missing}")
        
        # Check for duplicates
        if 'Well' in data.columns:
            duplicates = data['Well'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate well identifiers")
        
        # Check for outliers in expression data
        if 'Expression_Fold_Change' in data.columns:
            expr_data = data['Expression_Fold_Change'].dropna()
            if len(expr_data) > 0:
                Q1, Q3 = expr_data.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = ((expr_data < Q1 - 3*IQR) | (expr_data > Q3 + 3*IQR)).sum()
                outlier_pct = outliers / len(expr_data) * 100
                if outlier_pct > 10:
                    warnings.append(f"High percentage of outliers in expression data: {outlier_pct:.1f}%")
        
        return errors, warnings
    
    def _has_chemical_data(self, data: pd.DataFrame) -> bool:
        """Check if dataset contains chemical descriptor data"""
        chem_cols = [col for col in self.optional_columns if col in data.columns]
        return len(chem_cols) > 0
    
    def _validate_chemical_descriptors(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate chemical descriptor data"""
        errors = []
        warnings = []
        
        # Check molecular weight ranges
        if 'MW' in data.columns:
            mw_data = data['MW'].dropna()
            if len(mw_data) > 0:
                if mw_data.min() < 50 or mw_data.max() > 2000:
                    warnings.append(f"Unusual MW range: {mw_data.min():.1f} - {mw_data.max():.1f}")
        
        # Check LogP ranges
        if 'LogP' in data.columns:
            logp_data = data['LogP'].dropna()
            if len(logp_data) > 0:
                if logp_data.min() < -5 or logp_data.max() > 10:
                    warnings.append(f"Unusual LogP range: {logp_data.min():.1f} - {logp_data.max():.1f}")
        
        # Check for negative values in descriptors that should be positive
        positive_descriptors = ['MW', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'AromaticRings']
        for desc in positive_descriptors:
            if desc in data.columns:
                negative_values = (data[desc] < 0).sum()
                if negative_values > 0:
                    errors.append(f"Found {negative_values} negative values in {desc}")
        
        return errors, warnings
    
    def _validate_controls(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate control compounds"""
        errors = []
        warnings = []
        
        if 'Compound' not in data.columns:
            return errors, warnings
        
        # Check for presence of controls
        present_controls = [ctrl for ctrl in self.control_types if ctrl in data['Compound'].values]
        missing_controls = [ctrl for ctrl in self.control_types if ctrl not in present_controls]
        
        if missing_controls:
            warnings.append(f"Missing control types: {missing_controls}")
        
        # Check control sample sizes
        for ctrl in present_controls:
            ctrl_count = (data['Compound'] == ctrl).sum()
            if ctrl_count < 3:
                warnings.append(f"Low sample size for {ctrl}: {ctrl_count} wells")
        
        # Check control variability
        if 'Expression_Fold_Change' in data.columns:
            for ctrl in present_controls:
                ctrl_data = data[data['Compound'] == ctrl]['Expression_Fold_Change']
                if len(ctrl_data) > 1:
                    cv = ctrl_data.std() / ctrl_data.mean() * 100
                    if cv > 25:
                        warnings.append(f"High variability in {ctrl} controls: CV = {cv:.1f}%")
        
        return errors, warnings
    
    def _validate_statistics(self, data: pd.DataFrame) -> List[str]:
        """Validate statistical properties"""
        warnings = []
        
        if 'Expression_Fold_Change' not in data.columns:
            return warnings
        
        expr_data = data['Expression_Fold_Change'].dropna()
        if len(expr_data) == 0:
            warnings.append("No valid expression data found")
            return warnings
        
        # Check for sufficient sample size
        if len(expr_data) < 50:
            warnings.append(f"Small sample size for statistical analysis: {len(expr_data)} wells")
        
        # Check distribution properties
        skewness = expr_data.skew()
        if abs(skewness) > 2:
            warnings.append(f"Highly skewed expression distribution: skewness = {skewness:.2f}")
        
        # Check for constant values
        if expr_data.nunique() == 1:
            warnings.append("All expression values are identical")
        elif expr_data.nunique() < 10:
            warnings.append(f"Very few unique expression values: {expr_data.nunique()}")
        
        return warnings
    
    def _generate_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate dataset metadata"""
        metadata = {
            'n_wells': len(data),
            'n_compounds': data['Compound'].nunique() if 'Compound' in data.columns else 0,
            'columns': list(data.columns),
            'missing_data_pct': (data.isnull().sum().sum() / data.size * 100),
            'has_chemical_descriptors': self._has_chemical_data(data)
        }
        
        if 'Expression_Fold_Change' in data.columns:
            expr_data = data['Expression_Fold_Change'].dropna()
            if len(expr_data) > 0:
                metadata.update({
                    'expression_stats': {
                        'mean': float(expr_data.mean()),
                        'std': float(expr_data.std()),
                        'min': float(expr_data.min()),
                        'max': float(expr_data.max()),
                        'median': float(expr_data.median())
                    }
                })
        
        return metadata

class CacheManager:
    """Intelligent caching for expensive operations"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    timestamp REAL,
                    ttl REAL
                )
            """)
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function signature"""
        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data, timestamp, ttl FROM cache WHERE key = ?", (key,)
                )
                result = cursor.fetchone()
                
                if result is None:
                    return None
                
                data_blob, timestamp, ttl = result
                
                # Check if cache entry has expired
                if ttl > 0 and time.time() - timestamp > ttl:
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    return None
                
                return pickle.loads(data_blob)
                
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: float = 0) -> bool:
        """Store item in cache"""
        try:
            data_blob = pickle.dumps(value)
            timestamp = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, data, timestamp, ttl) VALUES (?, ?, ?, ?)",
                    (key, data_blob, timestamp, ttl)
                )
            return True
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {str(e)}")
            return False
    
    def clear_expired(self):
        """Clear expired cache entries"""
        try:
            current_time = time.time()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM cache WHERE ttl > 0 AND ? - timestamp > ttl",
                    (current_time,)
                )
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {str(e)}")

def cached(ttl: float = 3600):
    """Decorator for caching function results"""
    cache_manager = CacheManager()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_manager._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Compute result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(cache_key, result, ttl)
            return result
            
        return wrapper
    return decorator

class PerformanceMonitor:
    """Monitor and profile application performance"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_operation(self, operation_name: str):
        """Start timing an operation"""
        self.start_times[operation_name] = time.time()
    
    def end_operation(self, operation_name: str):
        """End timing an operation and record metrics"""
        if operation_name in self.start_times:
            duration = time.time() - self.start_times[operation_name]
            
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            
            self.metrics[operation_name].append(duration)
            del self.start_times[operation_name]
            
            logger.debug(f"Operation '{operation_name}' completed in {duration:.2f}s")
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations"""
        self.start_operation(operation_name)
        try:
            yield
        finally:
            self.end_operation(operation_name)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary"""
        summary = {}
        
        for operation, durations in self.metrics.items():
            if durations:
                summary[operation] = {
                    'count': len(durations),
                    'total_time': sum(durations),
                    'avg_time': np.mean(durations),
                    'min_time': min(durations),
                    'max_time': max(durations),
                    'std_time': np.std(durations)
                }
        
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_times.clear()

class AdvancedErrorHandler:
    """Advanced error handling with context and recovery strategies"""
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {}
    
    def register_recovery_strategy(self, error_type: type, strategy: Callable):
        """Register a recovery strategy for specific error types"""
        self.recovery_strategies[error_type] = strategy
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Any:
        """Handle error with context and attempt recovery"""
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'timestamp': time.time(),
            'context': context or {}
        }
        
        self.error_history.append(error_info)
        logger.error(f"Error handled: {error_info}")
        
        # Attempt recovery
        error_type = type(error)
        if error_type in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[error_type](error, context)
                logger.info(f"Recovery strategy successful for {error_type.__name__}")
                return recovery_result
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {str(recovery_error)}")
        
        # Re-raise if no recovery strategy or recovery failed
        raise error

# Unit Tests
import unittest
from unittest.mock import Mock, patch

class TestAdvancedDataValidator(unittest.TestCase):
    """Unit tests for AdvancedDataValidator"""
    
    def setUp(self):
        self.validator = AdvancedDataValidator()
    
    def test_empty_dataset(self):
        """Test validation of empty dataset"""
        empty_df = pd.DataFrame()
        result = self.validator.validate_dataset(empty_df)
        
        self.assertFalse(result.is_valid)
        self.assertIn("Dataset is empty", result.errors)
    
    def test_missing_required_columns(self):
        """Test validation with missing required columns"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = self.validator.validate_dataset(df)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Missing required columns" in error for error in result.errors))
    
    def test_valid_dataset(self):
        """Test validation of valid dataset"""
        df = pd.DataFrame({
            'Well': ['A1', 'A2', 'A3'],
            'Compound': ['Comp1', 'Comp2', 'Control+'],
            'Expression_Fold_Change': [1.5, 2.0, 3.0]
        })
        result = self.validator.validate_dataset(df)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_chemical_descriptor_validation(self):
        """Test chemical descriptor validation"""
        df = pd.DataFrame({
            'Well': ['A1', 'A2'],
            'Compound': ['Comp1', 'Comp2'],
            'Expression_Fold_Change': [1.5, 2.0],
            'MW': [300, -50]  # Invalid negative MW
        })
        result = self.validator.validate_dataset(df)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any("negative values in MW" in error for error in result.errors))

class TestCacheManager(unittest.TestCase):
    """Unit tests for CacheManager"""
    
    def setUp(self):
        self.cache = CacheManager(cache_dir="test_cache")
    
    def tearDown(self):
        # Cleanup test cache
        import shutil
        shutil.rmtree("test_cache", ignore_errors=True)
    
    def test_cache_set_get(self):
        """Test basic cache set and get operations"""
        key = "test_key"
        value = {"data": "test_value"}
        
        # Set cache
        success = self.cache.set(key, value)
        self.assertTrue(success)
        
        # Get from cache
        retrieved = self.cache.get(key)
        self.assertEqual(retrieved, value)
    
    def test_cache_expiry(self):
        """Test cache expiry functionality"""
        key = "expiring_key"
        value = "expiring_value"
        
        # Set with very short TTL
        self.cache.set(key, value, ttl=0.1)
        
        # Should be available immediately
        self.assertEqual(self.cache.get(key), value)
        
        # Wait for expiry
        time.sleep(0.2)
        
        # Should be expired
        self.assertIsNone(self.cache.get(key))

class TestPerformanceMonitor(unittest.TestCase):
    """Unit tests for PerformanceMonitor"""
    
    def setUp(self):
        self.monitor = PerformanceMonitor()
    
    def test_operation_timing(self):
        """Test operation timing functionality"""
        operation_name = "test_operation"
        
        with self.monitor.monitor_operation(operation_name):
            time.sleep(0.1)  # Simulate work
        
        summary = self.monitor.get_summary()
        self.assertIn(operation_name, summary)
        self.assertEqual(summary[operation_name]['count'], 1)
        self.assertGreaterEqual(summary[operation_name]['avg_time'], 0.1)

# Performance benchmarking utilities
class BenchmarkSuite:
    """Comprehensive benchmarking for HTS operations"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_data_loading(self, file_path: str, n_iterations: int = 5):
        """Benchmark data loading performance"""
        times = []
        
        for _ in range(n_iterations):
            start_time = time.time()
            try:
                pd.read_csv(file_path)
                times.append(time.time() - start_time)
            except Exception as e:
                logger.error(f"Data loading benchmark failed: {str(e)}")
                return None
        
        self.results['data_loading'] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': min(times),
            'max_time': max(times)
        }
        
        return self.results['data_loading']
    
    def benchmark_ml_training(self, X: pd.DataFrame, y: pd.Series, 
                            model_class, n_iterations: int = 3):
        """Benchmark ML model training performance"""
        times = []
        
        for _ in range(n_iterations):
            start_time = time.time()
            try:
                model = model_class()
                model.fit(X, y)
                times.append(time.time() - start_time)
            except Exception as e:
                logger.error(f"ML training benchmark failed: {str(e)}")
                return None
        
        self.results['ml_training'] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': min(times),
            'max_time': max(times)
        }
        
        return self.results['ml_training']

# Example usage and integration
if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Example performance monitoring
    monitor = PerformanceMonitor()
    
    with monitor.monitor_operation("data_processing"):
        # Simulate data processing
        data = pd.DataFrame(np.random.randn(1000, 10))
        processed_data = data.fillna(0)
        time.sleep(0.1)
    
    print("Performance Summary:")
    print(json.dumps(monitor.get_summary(), indent=2))