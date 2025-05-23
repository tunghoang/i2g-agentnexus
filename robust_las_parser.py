"""
robust_las_parser.py - Robust LAS file parsing module for A2A+MCP+LangChain architecture

This module provides robust parsing capabilities for LAS files, handling various
edge cases and format issues that may cause standard parsers to fail.
"""

import os
import sys
import json
import traceback
import numpy as np
import pandas as pd
import lasio
from typing import Tuple, Dict, List, Any, Optional, Union

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class RobustLASFile:
    """
    A robust representation of LAS file data that can be created from either
    a standard lasio object or from direct parsing when lasio fails.
    """

    def __init__(self):
        """Initialize an empty LAS file object"""
        self.well_info = {}
        self.curves = []
        self.data = None
        self.df = None
        self.index = []
        self.index_unit = "m"  # Default unit
        self.version = {}
        self.source_file = None
        self.parsing_method = None

    @classmethod
    def from_lasio(cls, las_obj, file_path=None):
        """
        Create from a standard lasio object

        Args:
            las_obj: A lasio LASFile object
            file_path: Optional source file path

        Returns:
            RobustLASFile: A new LAS file object
        """
        obj = cls()
        obj.source_file = file_path
        obj.parsing_method = "lasio"

        # Copy well info
        obj.well_info = {item.mnemonic: item.value for item in las_obj.well}

        # Copy curve info
        obj.curves = [{
            "mnemonic": curve.mnemonic,
            "unit": curve.unit,
            "descr": curve.descr,
            "data": curve.data
        } for curve in las_obj.curves]

        # Copy index (depth) info
        obj.index = las_obj.index
        obj.index_unit = las_obj.index_unit

        # Copy version info
        obj.version = {item.mnemonic: item.value for item in las_obj.version}

        # Copy data
        obj.data = las_obj.data

        # Create dataframe
        if hasattr(las_obj, "df"):
            obj.df = las_obj.df()
        else:
            # Create dataframe manually
            data_dict = {}
            for curve in obj.curves:
                data_dict[curve["mnemonic"]] = curve["data"]
            obj.df = pd.DataFrame(data_dict)

        return obj

    @classmethod
    def from_direct_parsing(cls, file_path):
        """
        Create by directly parsing a LAS file

        Args:
            file_path: Path to the LAS file

        Returns:
            RobustLASFile: A new LAS file object
        """
        obj = cls()
        obj.source_file = file_path
        obj.parsing_method = "direct"

        # Read the file content
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Process header sections
        section = "version"
        obj.version = {"VERS": "2.0", "WRAP": "NO"}  # Default values

        # Extract well info and curve info
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check for section markers
            if line.startswith('~'):
                if 'V' in line.upper():
                    section = "version"
                elif 'W' in line.upper():
                    section = "well"
                elif 'C' in line.upper():
                    section = "curve"
                elif 'P' in line.upper():
                    section = "parameter"
                elif 'O' in line.upper():
                    section = "other"
                elif 'A' in line.upper():
                    # Found data section
                    # The next non-empty, non-comment line should be column names
                    column_names = None
                    data_start_line = None

                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if next_line and not next_line.startswith('#'):
                            column_names = next_line.split()
                            data_start_line = j + 1
                            break

                    if column_names and data_start_line:
                        # Extract data
                        data_lines = []
                        for j in range(data_start_line, len(lines)):
                            data_line = lines[j].strip()
                            if not data_line or data_line.startswith('#'):
                                continue

                            values = data_line.split()
                            if len(values) == len(column_names):
                                # Try to convert to numeric
                                try:
                                    numeric_values = [float(v) for v in values]
                                    data_lines.append(numeric_values)
                                except ValueError:
                                    print(f"Warning: Skipping non-numeric line: {data_line}")

                        # Create dataframe
                        if data_lines:
                            obj.df = pd.DataFrame(data_lines, columns=column_names)

                            # Set index
                            obj.index = obj.df.iloc[:, 0].values

                            # Create curve info
                            obj.curves = []
                            for col in obj.df.columns:
                                obj.curves.append({
                                    "mnemonic": col,
                                    "unit": "unknown",
                                    "descr": col,
                                    "data": obj.df[col].values
                                })

                            # Create data array
                            obj.data = obj.df.values
                continue

            # Process well info and version info from headers
            if ':' in line and not line.startswith('#'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    header_part = parts[0].strip()
                    value_part = parts[1].strip()

                    # Extract the mnemonic and unit
                    if '.' in header_part:
                        mnemonic = header_part.split('.')[0].strip()

                        if section == "well":
                            obj.well_info[mnemonic] = value_part
                        elif section == "version":
                            obj.version[mnemonic] = value_part

        # If we didn't find a data section or failed to create a dataframe
        if obj.df is None:
            obj.df = pd.DataFrame()

        return obj

    def get_curve_data(self, mnemonic):
        """
        Get data for a specific curve

        Args:
            mnemonic: The curve mnemonic

        Returns:
            numpy.ndarray: The curve data
        """
        if self.df is not None and mnemonic in self.df.columns:
            return self.df[mnemonic].values

        for curve in self.curves:
            if curve["mnemonic"] == mnemonic:
                return curve["data"]

        return np.array([])

    def curve_exists(self, mnemonic):
        """Check if a curve exists"""
        if self.df is not None:
            return mnemonic in self.df.columns

        for curve in self.curves:
            if curve["mnemonic"] == mnemonic:
                return True

        return False

    def get_curve_names(self):
        """Get all curve names"""
        if self.df is not None:
            return list(self.df.columns)

        return [curve["mnemonic"] for curve in self.curves]

    def get_depth_range(self):
        """Get the depth range"""
        if len(self.index) > 0:
            return (float(self.index[0]), float(self.index[-1]))

        if self.df is not None and len(self.df) > 0:
            # Assume first column is depth
            return (float(self.df.iloc[0, 0]), float(self.df.iloc[-1, 0]))

        return (0, 0)


def load_las_file(file_path: str) -> Tuple[Optional[RobustLASFile], Optional[str]]:
    """
    Load a LAS file using both standard and robust methods

    Args:
        file_path: Path to the LAS file

    Returns:
        tuple: (RobustLASFile object, error message or None)
    """
    print(f"Loading LAS file: {file_path}")

    # Try standard parsing first
    try:
        las = lasio.read(file_path)

        # Check if data was parsed correctly (numeric)
        try:
            # Check if at least one numeric column exists
            is_numeric = False
            for curve in las.curves:
                if np.issubdtype(curve.data.dtype, np.number):
                    is_numeric = True
                    break

            if is_numeric:
                # Standard parsing worked
                print("Standard parsing successful - data is numeric")
                return RobustLASFile.from_lasio(las, file_path), None
            else:
                print("Standard parsing produced non-numeric data, trying direct parsing")
        except Exception as e:
            print(f"Error checking numeric data: {str(e)}")
    except Exception as e:
        print(f"Standard parsing failed: {str(e)}")

    # If standard parsing failed or produced non-numeric data, try direct parsing
    try:
        print("Attempting direct parsing")
        las = RobustLASFile.from_direct_parsing(file_path)

        # Check if we got any data
        if las.df is not None and len(las.df) > 0:
            print(f"Direct parsing successful - found {len(las.df)} rows")
            return las, None
        else:
            return None, "Direct parsing failed to extract any data"
    except Exception as e:
        error_details = traceback.format_exc()
        return None, f"All parsing methods failed: {str(e)}\n{error_details}"


def analyze_curve(las: RobustLASFile, curve_name: str) -> Dict[str, Any]:
    """
    Analyze a curve in a LAS file

    Args:
        las: A RobustLASFile object
        curve_name: Name of the curve to analyze

    Returns:
        dict: Analysis results
    """
    if not las.curve_exists(curve_name):
        return {"error": f"Curve {curve_name} not found", "available_curves": las.get_curve_names()}

    # Get the curve data
    data = las.get_curve_data(curve_name)

    # Convert to pandas Series for easier analysis
    s = pd.Series(data)

    # Filter out NaN values
    clean_data = s.dropna()

    # Calculate statistics
    analysis = {
        "curve": curve_name,
        "min": float(clean_data.min()) if len(clean_data) > 0 else None,
        "max": float(clean_data.max()) if len(clean_data) > 0 else None,
        "mean": float(clean_data.mean()) if len(clean_data) > 0 else None,
        "median": float(clean_data.median()) if len(clean_data) > 0 else None,
        "std_dev": float(clean_data.std()) if len(clean_data) > 0 else None,
        "percentiles": {
            "p10": float(np.percentile(clean_data, 10)) if len(clean_data) > 0 else None,
            "p50": float(np.percentile(clean_data, 50)) if len(clean_data) > 0 else None,
            "p90": float(np.percentile(clean_data, 90)) if len(clean_data) > 0 else None
        },
        "total_points": len(data),
        "valid_points": len(clean_data),
        "null_count": len(data) - len(clean_data),
        "null_percentage": round((len(data) - len(clean_data)) / len(data) * 100, 2) if len(data) > 0 else 0
    }

    return analysis


def perform_quality_check(las: RobustLASFile) -> Dict[str, Any]:
    """
    Perform quality checks on a LAS file

    Args:
        las: A RobustLASFile object

    Returns:
        dict: QC results
    """
    qc_results = {
        "file_info": {
            "filename": os.path.basename(las.source_file) if las.source_file else "Unknown",
            "parsing_method": las.parsing_method,
            "curve_count": len(las.get_curve_names()),
            "data_points": len(las.df) if las.df is not None else 0
        },
        "issues": [],
        "curve_issues": {}
    }

    # Check well headers
    missing_headers = []
    important_headers = ["WELL", "API", "STRT", "STOP", "STEP", "NULL"]
    for header in important_headers:
        if header not in las.well_info or las.well_info[header] is None:
            missing_headers.append(header)

    if missing_headers:
        qc_results["issues"].append({
            "severity": "warning",
            "message": f"Missing important header(s): {', '.join(missing_headers)}"
        })

    # Check each curve for issues
    for curve_name in las.get_curve_names():
        curve_issues = []
        data = las.get_curve_data(curve_name)

        # Convert to pandas Series for easier analysis
        s = pd.Series(data)

        # Check for null values
        null_count = s.isna().sum()
        if null_count > 0:
            null_percent = null_count / len(s) * 100
            severity = "info" if null_percent < 5 else ("warning" if null_percent < 30 else "error")
            curve_issues.append({
                "severity": severity,
                "message": f"{null_percent:.1f}% null values ({null_count} out of {len(s)})"
            })

        # Check for constant values
        clean_data = s.dropna()
        if len(clean_data) > 0:
            unique_values = clean_data.nunique()
            if unique_values < 3:
                curve_issues.append({
                    "severity": "warning",
                    "message": f"Constant or near-constant values (only {unique_values} unique values)"
                })

        # Check for outliers
        if len(clean_data) > 10:
            q1, q3 = clean_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                outlier_percent = len(outliers) / len(clean_data) * 100
                if outlier_percent > 1:
                    curve_issues.append({
                        "severity": "info",
                        "message": f"{outlier_percent:.1f}% outlier values detected"
                    })

        # Add curve issues if any found
        if curve_issues:
            qc_results["curve_issues"][curve_name] = curve_issues

    # Overall quality assessment
    error_count = sum(1 for issue in qc_results["issues"] if issue["severity"] == "error")
    warning_count = sum(1 for issue in qc_results["issues"] if issue["severity"] == "warning")

    for curve, issues in qc_results["curve_issues"].items():
        error_count += sum(1 for issue in issues if issue["severity"] == "error")
        warning_count += sum(1 for issue in issues if issue["severity"] == "warning")

    if error_count > 0:
        qc_results["quality_rating"] = "Poor"
    elif warning_count > 3:
        qc_results["quality_rating"] = "Fair"
    elif warning_count > 0:
        qc_results["quality_rating"] = "Good"
    else:
        qc_results["quality_rating"] = "Excellent"

    return qc_results


def extract_metadata(las: RobustLASFile) -> Dict[str, Any]:
    """
    Extract metadata from a LAS file

    Args:
        las: A RobustLASFile object

    Returns:
        dict: Metadata
    """
    # Basic info
    metadata = {
        "well_name": las.well_info.get("WELL", "Unknown"),
        "api": las.well_info.get("API", "Unknown"),
        "uwi": las.well_info.get("UWI", "Unknown"),
        "company": las.well_info.get("COMP", "Unknown"),
        "field": las.well_info.get("FLD", "Unknown"),
        "location": {},
        "well_info": las.well_info,
        "parsing_method": las.parsing_method
    }

    # Location info
    location_keys = ["LATI", "LONG", "X", "Y", "SECT", "TOWN", "RANG", "CTRY", "STAT", "CNTY", "PROV"]
    for key in location_keys:
        if key in las.well_info and las.well_info[key]:
            metadata["location"][key] = las.well_info[key]

    # Depth info
    depth_range = las.get_depth_range()
    metadata["depth_info"] = {
        "start": depth_range[0],
        "end": depth_range[1],
        "unit": las.index_unit,
        "points": len(las.index) if hasattr(las.index, "__len__") else 0
    }

    # Curve info
    metadata["curves"] = []
    curve_names = las.get_curve_names()
    for i, name in enumerate(curve_names):
        if i < len(las.curves) and isinstance(las.curves[i], dict):
            metadata["curves"].append({
                "mnemonic": name,
                "unit": las.curves[i].get("unit", "unknown"),
                "description": las.curves[i].get("descr", name)
            })
        else:
            metadata["curves"].append({
                "mnemonic": name,
                "unit": "unknown",
                "description": name
            })

    return metadata


# Simple test code if run directly
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python robust_las_parser.py <las_file> [curve_name]")
        sys.exit(1)

    las_file = sys.argv[1]
    las_obj, error = load_las_file(las_file)

    if error:
        print(f"Error loading LAS file: {error}")
        sys.exit(1)

    print("\nMetadata:")
    metadata = extract_metadata(las_obj)
    print(json.dumps(metadata, indent=2, cls=NumpyJSONEncoder))

    if len(sys.argv) > 2:
        curve_name = sys.argv[2]
        print(f"\nAnalyzing curve: {curve_name}")
        analysis = analyze_curve(las_obj, curve_name)
        print(json.dumps(analysis, indent=2, cls=NumpyJSONEncoder))

    print("\nQuality Check:")
    qc = perform_quality_check(las_obj)
    print(json.dumps(qc, indent=2, cls=NumpyJSONEncoder))