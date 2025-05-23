"""
formation_evaluation.py - Advanced petrophysical analysis module for LAS files

This module provides functions for calculating key petrophysical properties
from well log data, including:
- Shale volume (Vshale) from gamma ray
- Effective porosity
- Water saturation
- Pay zone identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json
import traceback

# Import the robust LAS parser (adjust path if needed)
from robust_las_parser import load_las_file, RobustLASFile


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


def estimate_vshale(gr_values: np.ndarray, gr_clean: Optional[float] = None,
                    gr_shale: Optional[float] = None) -> np.ndarray:
    """
    Calculate shale volume (Vshale) from gamma ray log

    Args:
        gr_values: Gamma ray values array
        gr_clean: Gamma ray value in clean sand (if None, estimated from data)
        gr_shale: Gamma ray value in pure shale (if None, estimated from data)

    Returns:
        numpy.ndarray: Calculated Vshale values (0-1 range)
    """
    # Handle missing values
    if gr_values is None or len(gr_values) == 0:
        return np.array([])

    # Remove NaNs for statistical calculations
    clean_gr = gr_values[~np.isnan(gr_values)]
    if len(clean_gr) == 0:
        return np.full_like(gr_values, np.nan)

    # Auto-estimate GR clean and GR shale if not provided
    if gr_clean is None:
        # Use 5th percentile for clean sand GR
        gr_clean = np.percentile(clean_gr, 5)

    if gr_shale is None:
        # Use 95th percentile for shale GR
        gr_shale = np.percentile(clean_gr, 95)

    # Ensure gr_shale > gr_clean to avoid division by zero or negative values
    if gr_shale <= gr_clean:
        gr_shale = gr_clean + 1  # Avoid division by zero

    # Calculate linear Vshale (IGR - Gamma Ray Index)
    vsh_linear = np.full_like(gr_values, np.nan, dtype=float)
    valid_mask = ~np.isnan(gr_values)
    vsh_linear[valid_mask] = (gr_values[valid_mask] - gr_clean) / (gr_shale - gr_clean)

    # Clip values to 0-1 range
    vsh_linear = np.clip(vsh_linear, 0, 1)

    # Apply Larionov correction for Tertiary rocks
    # Vsh = 0.083 * (2^(3.7*IGR) - 1)
    vsh_corrected = 0.083 * (np.power(2, 3.7 * vsh_linear) - 1)

    # Clip again to ensure 0-1 range
    vsh_corrected = np.clip(vsh_corrected, 0, 1)

    return vsh_corrected


def calculate_porosity(density_values: np.ndarray,
                       matrix_density: float = 2.65,
                       fluid_density: float = 1.0) -> np.ndarray:
    """
    Calculate porosity from density log using the density porosity equation

    Args:
        density_values: Bulk density values array
        matrix_density: Density of the rock matrix (default: 2.65 g/cc for sandstone)
        fluid_density: Density of the fluid (default: 1.0 g/cc for water)

    Returns:
        numpy.ndarray: Calculated porosity values (0-1 range)
    """
    # Handle missing values
    if density_values is None or len(density_values) == 0:
        return np.array([])

    # Calculate density porosity
    porosity = np.full_like(density_values, np.nan, dtype=float)
    valid_mask = ~np.isnan(density_values)

    # Apply density porosity equation
    porosity[valid_mask] = (matrix_density - density_values[valid_mask]) / (matrix_density - fluid_density)

    # Clip to reasonable range (0-0.5 or 0-50%)
    porosity = np.clip(porosity, 0, 0.5)

    return porosity


def calculate_effective_porosity(total_porosity: np.ndarray,
                                 vshale: np.ndarray,
                                 shale_porosity: float = 0.1) -> np.ndarray:
    """
    Calculate effective porosity by correcting for shale content

    Args:
        total_porosity: Total porosity values array
        vshale: Shale volume fraction array
        shale_porosity: Typical porosity of shale (default: 0.1)

    Returns:
        numpy.ndarray: Calculated effective porosity values (0-1 range)
    """
    # Handle missing values
    if total_porosity is None or vshale is None:
        return np.array([])

    if len(total_porosity) != len(vshale):
        raise ValueError(f"Arrays must be same length, got {len(total_porosity)} and {len(vshale)}")

    # Calculate effective porosity
    phie = np.full_like(total_porosity, np.nan, dtype=float)
    valid_mask = ~np.isnan(total_porosity) & ~np.isnan(vshale)

    # Apply effective porosity equation
    phie[valid_mask] = total_porosity[valid_mask] - (vshale[valid_mask] * shale_porosity)

    # Clip to reasonable range (0-0.5 or 0-50%)
    phie = np.clip(phie, 0, 0.5)

    return phie


def calculate_water_saturation(resistivity: np.ndarray,
                               porosity: np.ndarray,
                               rw: float = 0.1,
                               a: float = 1.0,
                               m: float = 2.0,
                               n: float = 2.0) -> np.ndarray:
    """
    Calculate water saturation using Archie's equation

    Args:
        resistivity: Deep resistivity values array
        porosity: Effective porosity values array
        rw: Formation water resistivity (default: 0.1 ohm-m)
        a: Tortuosity factor (default: 1.0)
        m: Cementation exponent (default: 2.0)
        n: Saturation exponent (default: 2.0)

    Returns:
        numpy.ndarray: Calculated water saturation values (0-1 range)
    """
    # Handle missing values
    if resistivity is None or porosity is None:
        return np.array([])

    if len(resistivity) != len(porosity):
        raise ValueError(f"Arrays must be same length, got {len(resistivity)} and {len(porosity)}")

    # Calculate water saturation
    sw = np.full_like(resistivity, np.nan, dtype=float)
    valid_mask = ~np.isnan(resistivity) & ~np.isnan(porosity) & (porosity > 0) & (resistivity > 0)

    # Apply Archie's equation
    F = a / np.power(porosity[valid_mask], m)  # Formation factor
    sw[valid_mask] = np.power((F * rw / resistivity[valid_mask]), (1 / n))

    # Clip to reasonable range (0-1 or 0-100%)
    sw = np.clip(sw, 0, 1)

    return sw


def identify_pay_zones(depth: np.ndarray,
                       vshale: np.ndarray,
                       porosity: np.ndarray,
                       sw: np.ndarray,
                       vsh_cutoff: float = 0.5,
                       porosity_cutoff: float = 0.1,
                       sw_cutoff: float = 0.7) -> List[Dict[str, Any]]:
    """
    Identify potential hydrocarbon pay zones based on cutoffs

    Args:
        depth: Depth values array
        vshale: Shale volume fraction array
        porosity: Effective porosity values array
        sw: Water saturation values array
        vsh_cutoff: Maximum acceptable Vshale (default: 0.5)
        porosity_cutoff: Minimum acceptable porosity (default: 0.1)
        sw_cutoff: Maximum acceptable water saturation (default: 0.7)

    Returns:
        List[Dict]: List of pay zones with start depth, end depth, and properties
    """
    # Handle missing values
    if depth is None or len(depth) == 0:
        return []

    # Ensure all arrays are same length
    if not (len(depth) == len(vshale) == len(porosity) == len(sw)):
        raise ValueError("All arrays must have the same length")

    # Create pay flag array (1 for pay, 0 for non-pay)
    pay_flag = np.zeros_like(depth)
    valid_mask = ~np.isnan(vshale) & ~np.isnan(porosity) & ~np.isnan(sw)

    # Apply cutoffs
    pay_conditions = (
            (vshale[valid_mask] <= vsh_cutoff) &
            (porosity[valid_mask] >= porosity_cutoff) &
            (sw[valid_mask] <= sw_cutoff)
    )

    # Set pay flag
    pay_flag[valid_mask] = np.where(pay_conditions, 1, 0)

    # Find pay zone boundaries
    pay_zones = []
    in_pay_zone = False
    start_idx = None

    for i in range(len(pay_flag)):
        if pay_flag[i] == 1 and not in_pay_zone:
            # Start of new pay zone
            in_pay_zone = True
            start_idx = i
        elif pay_flag[i] == 0 and in_pay_zone:
            # End of pay zone
            in_pay_zone = False
            end_idx = i - 1

            # Calculate zone properties
            zone_depth = depth[start_idx:end_idx + 1]
            zone_vsh = vshale[start_idx:end_idx + 1]
            zone_porosity = porosity[start_idx:end_idx + 1]
            zone_sw = sw[start_idx:end_idx + 1]

            # Only include zones with minimum thickness (e.g., 0.5m)
            thickness = zone_depth[-1] - zone_depth[0]
            if thickness >= 0.5:  # Minimum 0.5m thickness
                pay_zones.append({
                    "start_depth": float(zone_depth[0]),
                    "end_depth": float(zone_depth[-1]),
                    "thickness": float(thickness),
                    "avg_vshale": float(np.mean(zone_vsh)),
                    "avg_porosity": float(np.mean(zone_porosity)),
                    "avg_sw": float(np.mean(zone_sw)),
                    "hc_saturation": float(1 - np.mean(zone_sw)),  # Hydrocarbon saturation
                    "net_pay": float(thickness)
                })

    # Handle case where file ends while still in pay zone
    if in_pay_zone:
        end_idx = len(pay_flag) - 1

        # Calculate zone properties
        zone_depth = depth[start_idx:end_idx + 1]
        zone_vsh = vshale[start_idx:end_idx + 1]
        zone_porosity = porosity[start_idx:end_idx + 1]
        zone_sw = sw[start_idx:end_idx + 1]

        # Only include zones with minimum thickness (e.g., 0.5m)
        thickness = zone_depth[-1] - zone_depth[0]
        if thickness >= 0.5:  # Minimum 0.5m thickness
            pay_zones.append({
                "start_depth": float(zone_depth[0]),
                "end_depth": float(zone_depth[-1]),
                "thickness": float(thickness),
                "avg_vshale": float(np.mean(zone_vsh)),
                "avg_porosity": float(np.mean(zone_porosity)),
                "avg_sw": float(np.mean(zone_sw)),
                "hc_saturation": float(1 - np.mean(zone_sw)),  # Hydrocarbon saturation
                "net_pay": float(thickness)
            })

    return pay_zones


def evaluate_formation(las_file: Union[str, RobustLASFile],
                       gr_curve: str = "GR",
                       density_curve: str = "RHOB",
                       resistivity_curve: str = "RT",
                       neutron_curve: str = "NPHI",
                       matrix_density: float = 2.65,
                       fluid_density: float = 1.0,
                       rw: float = 0.1,
                       vsh_cutoff: float = 0.5,
                       porosity_cutoff: float = 0.1,
                       sw_cutoff: float = 0.7) -> Dict[str, Any]:
    """
    Perform comprehensive formation evaluation on a LAS file

    Args:
        las_file: Path to LAS file or RobustLASFile object
        gr_curve: Name of gamma ray curve (default: "GR")
        density_curve: Name of density curve (default: "RHOB")
        resistivity_curve: Name of resistivity curve (default: "RT")
        neutron_curve: Name of neutron porosity curve (default: "NPHI")
        matrix_density: Density of the rock matrix (default: 2.65 g/cc for sandstone)
        fluid_density: Density of the fluid (default: 1.0 g/cc for water)
        rw: Formation water resistivity (default: 0.1 ohm-m)
        vsh_cutoff: Maximum acceptable Vshale for pay (default: 0.5)
        porosity_cutoff: Minimum acceptable porosity for pay (default: 0.1)
        sw_cutoff: Maximum acceptable water saturation for pay (default: 0.7)

    Returns:
        Dict: Evaluation results including Vshale, porosity, water saturation, and pay zones
    """
    # Load LAS file if string path is provided
    if isinstance(las_file, str):
        las, error = load_las_file(las_file)
        if error:
            return {"error": error}
    else:
        las = las_file

    # Extract depth and curves
    depth = las.index

    # Check for required curves
    required_curves = [gr_curve, density_curve, resistivity_curve]
    missing_curves = [curve for curve in required_curves if not las.curve_exists(curve)]

    if missing_curves:
        return {
            "error": f"Missing required curves: {', '.join(missing_curves)}",
            "available_curves": las.get_curve_names()
        }

    # Extract curve data
    gr_data = las.get_curve_data(gr_curve)
    density_data = las.get_curve_data(density_curve)
    resistivity_data = las.get_curve_data(resistivity_curve)

    # Extract neutron data if available
    neutron_data = None
    if neutron_curve and las.curve_exists(neutron_curve):
        neutron_data = las.get_curve_data(neutron_curve)

    # Perform calculations
    vshale = estimate_vshale(gr_data)
    density_porosity = calculate_porosity(density_data, matrix_density, fluid_density)

    # Use density porosity as total porosity, or average with neutron if available
    if neutron_data is not None:
        # For limestone scale (typical neutron log), convert neutron values if needed
        # Assuming neutron values are already in fraction, not percentage
        neutron_porosity = np.clip(neutron_data, 0, 0.5)
        # Calculate total porosity as average of density and neutron
        total_porosity = (density_porosity + neutron_porosity) / 2
    else:
        total_porosity = density_porosity

    # Calculate effective porosity and water saturation
    effective_porosity = calculate_effective_porosity(total_porosity, vshale)
    sw = calculate_water_saturation(resistivity_data, effective_porosity, rw)

    # Identify potential pay zones
    pay_zones = identify_pay_zones(
        depth, vshale, effective_porosity, sw,
        vsh_cutoff, porosity_cutoff, sw_cutoff
    )

    # Calculate net pay
    net_pay = sum(zone["thickness"] for zone in pay_zones)

    # Calculate average reservoir properties
    valid_mask = ~np.isnan(vshale) & ~np.isnan(effective_porosity) & ~np.isnan(sw)
    if np.any(valid_mask):
        avg_vshale = float(np.mean(vshale[valid_mask]))
        avg_porosity = float(np.mean(effective_porosity[valid_mask]))
        avg_sw = float(np.mean(sw[valid_mask]))
    else:
        avg_vshale = None
        avg_porosity = None
        avg_sw = None

    # Build result dictionary
    result = {
        "well_name": las.well_info.get("WELL", "Unknown"),
        "depth_range": list(las.get_depth_range()),
        "curves_used": {
            "gamma_ray": gr_curve,
            "density": density_curve,
            "resistivity": resistivity_curve,
            "neutron": neutron_curve if neutron_data is not None else None
        },
        "parameters": {
            "matrix_density": matrix_density,
            "fluid_density": fluid_density,
            "formation_water_resistivity": rw,
            "vshale_cutoff": vsh_cutoff,
            "porosity_cutoff": porosity_cutoff,
            "sw_cutoff": sw_cutoff
        },
        "formation_properties": {
            "avg_vshale": avg_vshale,
            "avg_porosity": avg_porosity,
            "avg_sw": avg_sw,
            "avg_hc_saturation": float(1 - avg_sw) if avg_sw is not None else None
        },
        "pay_summary": {
            "net_pay": net_pay,
            "num_zones": len(pay_zones),
            "zones": pay_zones
        },
        "evaluation_quality": "Good" if not missing_curves else "Limited"
    }

    return result


# Additional function to create a summarized report for non-technical users
def create_formation_evaluation_summary(evaluation_result: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of formation evaluation results

    Args:
        evaluation_result: Result dictionary from evaluate_formation function

    Returns:
        str: Human-readable summary
    """
    if "error" in evaluation_result:
        return f"Error in formation evaluation: {evaluation_result['error']}"

    # Extract key information
    well_name = evaluation_result["well_name"]
    depth_range = evaluation_result["depth_range"]
    formation_props = evaluation_result["formation_properties"]
    pay_summary = evaluation_result["pay_summary"]

    # Create summary text
    summary = f"Formation Evaluation Summary for {well_name}\n"
    summary += f"=========================================\n\n"

    summary += f"Depth Range: {depth_range[0]:.1f} - {depth_range[1]:.1f} m\n\n"

    summary += "Formation Properties:\n"
    summary += f"- Average Shale Volume: {formation_props['avg_vshale'] * 100:.1f}%\n"
    summary += f"- Average Effective Porosity: {formation_props['avg_porosity'] * 100:.1f}%\n"
    summary += f"- Average Water Saturation: {formation_props['avg_sw'] * 100:.1f}%\n"
    summary += f"- Average Hydrocarbon Saturation: {formation_props['avg_hc_saturation'] * 100:.1f}%\n\n"

    summary += "Pay Zone Summary:\n"
    summary += f"- Net Pay: {pay_summary['net_pay']:.2f} m\n"
    summary += f"- Number of Pay Zones: {pay_summary['num_zones']}\n\n"

    if pay_summary['num_zones'] > 0:
        summary += "Detailed Pay Zones:\n"
        for i, zone in enumerate(pay_summary['zones'], 1):
            summary += f"Zone {i}: {zone['start_depth']:.1f} - {zone['end_depth']:.1f} m "
            summary += f"({zone['thickness']:.2f} m thick), "
            summary += f"Porosity: {zone['avg_porosity'] * 100:.1f}%, "
            summary += f"HC Saturation: {zone['hc_saturation'] * 100:.1f}%\n"
    else:
        summary += "No pay zones identified based on the specified cutoffs.\n"

    summary += "\nRecommendations:\n"
    if pay_summary['net_pay'] > 5:
        summary += "- The well shows significant net pay and good reservoir properties.\n"
        summary += "- Consider production testing or completion in the identified pay zones.\n"
    elif pay_summary['net_pay'] > 1:
        summary += "- The well shows moderate pay potential.\n"
        summary += "- Additional evaluation recommended before commitment to completion.\n"
    else:
        summary += "- Limited pay potential identified in this well.\n"
        summary += "- Consider additional logging or testing to verify results.\n"

    return summary


# Test code if run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python formation_evaluation.py <las_file>")
        sys.exit(1)

    las_file = sys.argv[1]
    print(f"Evaluating formation for: {las_file}")

    # Evaluate formation
    result = evaluate_formation(las_file)

    # Print JSON result
    print("\nDetailed Results:")
    print(json.dumps(result, indent=2, cls=NumpyJSONEncoder))

    # Print human-readable summary
    print("\nSummary Report:")
    print(create_formation_evaluation_summary(result))