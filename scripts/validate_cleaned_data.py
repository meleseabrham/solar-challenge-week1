"""Central validation script for exported cleaned CSV files.

This script validates that all cleaned CSV files meet quality standards:
- Required columns present (GHI, DNI, DHI)
- No negative values for solar metrics
- Reasonable value ranges
- Timestamp validity (if present)
- Data completeness checks
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from app.utils import SOLAR_COLS, FILE_MAP, DATA_DIR


def validate_solar_metric(series: pd.Series, metric_name: str) -> Tuple[bool, List[str]]:
    """Validate a single solar metric column.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    is_valid = True
    
    # Check for negative values
    negative_count = (series < 0).sum()
    if negative_count > 0:
        errors.append(f"{metric_name}: Found {negative_count} negative values")
        is_valid = False
    
    # Check for reasonable maximum values
    max_values = {
        "GHI": 1500,
        "DNI": 1100,
        "DHI": 400,
    }
    if metric_name in max_values:
        max_reasonable = max_values[metric_name]
        excessive_count = (series > max_reasonable).sum()
        if excessive_count > 0:
            errors.append(
                f"{metric_name}: Found {excessive_count} values exceeding "
                f"physical maximum ({max_reasonable} W/m²)"
            )
            # Warning, not necessarily invalid
    
    # Check for excessive missing values (>50%)
    missing_pct = series.isna().sum() / len(series) * 100
    if missing_pct > 50:
        errors.append(
            f"{metric_name}: {missing_pct:.1f}% missing values (threshold: 50%)"
        )
        is_valid = False
    
    return is_valid, errors


def validate_timestamp(series: pd.Series) -> Tuple[bool, List[str]]:
    """Validate timestamp column."""
    errors = []
    is_valid = True
    
    if not pd.api.types.is_datetime64_any_dtype(series):
        errors.append("Timestamp column is not datetime type")
        is_valid = False
    
    # Check for excessive missing timestamps
    missing_pct = series.isna().sum() / len(series) * 100
    if missing_pct > 10:
        errors.append(f"Timestamp: {missing_pct:.1f}% missing (threshold: 10%)")
        is_valid = False
    
    return is_valid, errors


def validate_cleaned_file(file_path: Path, country: str) -> Tuple[bool, List[str]]:
    """Validate a single cleaned CSV file.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    is_valid = True
    
    if not file_path.exists():
        return False, [f"File does not exist: {file_path}"]
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return False, [f"Failed to read CSV: {str(e)}"]
    
    # Check required columns
    missing_cols = [col for col in SOLAR_COLS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        is_valid = False
    
    # Validate each solar metric
    for metric in SOLAR_COLS:
        if metric in df.columns:
            metric_valid, metric_errors = validate_solar_metric(df[metric], metric)
            errors.extend(metric_errors)
            if not metric_valid:
                is_valid = False
    
    # Validate timestamp if present
    if "Timestamp" in df.columns:
        ts_valid, ts_errors = validate_timestamp(df["Timestamp"])
        errors.extend(ts_errors)
        if not ts_valid:
            is_valid = False
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        errors.append(f"Found {duplicate_count} duplicate rows")
        # Warning, not necessarily invalid
    
    # Check minimum data size
    if len(df) < 100:
        errors.append(f"Dataset too small: {len(df)} rows (minimum: 100)")
        is_valid = False
    
    return is_valid, errors


def validate_all_cleaned_files() -> Dict[str, Tuple[bool, List[str]]]:
    """Validate all cleaned CSV files.
    
    Returns:
        Dictionary mapping country names to (is_valid, errors) tuples
    """
    results = {}
    
    for country, config in FILE_MAP.items():
        clean_path = DATA_DIR / config["clean"]
        print(f"\n{'='*60}")
        print(f"Validating {country}: {clean_path.name}")
        print(f"{'='*60}")
        
        is_valid, errors = validate_cleaned_file(clean_path, country)
        results[country] = (is_valid, errors)
        
        if is_valid:
            print(f"✓ {country}: VALID")
        else:
            print(f"✗ {country}: INVALID")
        
        if errors:
            print("\nIssues found:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("  No issues found.")
    
    return results


def main():
    """Main validation function."""
    print("="*60)
    print("CLEANED DATA VALIDATION REPORT")
    print("="*60)
    
    results = validate_all_cleaned_files()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
    total_count = len(results)
    
    print(f"Valid files: {valid_count}/{total_count}")
    
    if valid_count == total_count:
        print("\n✓ All cleaned files passed validation!")
        return 0
    else:
        print(f"\n✗ {total_count - valid_count} file(s) failed validation")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

