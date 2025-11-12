# Scripts Directory

This directory contains utility scripts for data validation and processing.

## validate_cleaned_data.py

Central validation script for exported cleaned CSV files.

### Usage

From the project root:
```bash
python solar-challenge-week1/scripts/validate_cleaned_data.py
```

Or from the scripts directory:
```bash
cd solar-challenge-week1/scripts
python validate_cleaned_data.py
```

### What it validates

- **Required columns**: Ensures GHI, DNI, and DHI columns are present
- **Negative values**: Flags any negative values in solar metrics
- **Physical limits**: Checks for values exceeding physical maximums:
  - GHI: 1500 W/m²
  - DNI: 1100 W/m²
  - DHI: 400 W/m²
- **Missing data**: Warns if >50% of values are missing for any metric
- **Timestamp validity**: Validates timestamp column format and completeness
- **Duplicates**: Reports duplicate rows
- **Data size**: Ensures minimum dataset size (100 rows)

### Output

The script provides:
- Per-file validation results
- Detailed error messages for any issues
- Summary report of all files

Exit code 0 indicates all files passed validation; exit code 1 indicates one or more files failed.

