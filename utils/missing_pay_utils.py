import json
import os
from enhanced_mcp_tools import enhanced_las_parser
import utils.excel_utils as excel_utils


def well_checklist(
    wells: list[str],
    wells_dir: str = "data/wells",
    marker_dir: str = "data/misc/Marker.xlsx",
):
    if not os.path.isdir(wells_dir):
        raise Exception(f"Directory {wells_dir} does not exist")

    well_names = [f.name for f in os.scandir(wells_dir) if f.is_dir()]
    if wells is not None and len(wells) > 0:
        well_names = [f for f in well_names if f in wells]
    well_names.sort()
    count = len(well_names)
    if count == 0:
        raise Exception(f"No wells found for {wells}")

    loai_gieng_result: list[str] = ["N/A"] * count
    ten_gian_result: list[str] = ["N/A"] * count
    log_result: list[str] = ["N/A"] * count
    devi_result: list[str] = ["N/A"] * count
    mudlog_result: list[str] = ["N/A"] * count
    marker_result: list[str] = ["N/A"] * count
    thu_via_result: list[str] = ["N/A"] * count
    plt_result: list[str] = ["N/A"] * count
    kqdvl_result: list[str] = ["N/A"] * count

    prod_df = excel_utils.parse_well_production()
    prod_cols = prod_df.columns
    DATE_COL = 1
    WELL_COL = 1
    RIG_COL = 4
    OIL_RATE_COL = 6
    WATER_INJ_COL = 14
    max_date_idx = prod_df.groupby([prod_cols[WELL_COL]])[prod_cols[DATE_COL]].idxmax()
    prod_df = prod_df.loc[max_date_idx]
    oil_rate = prod_df[prod_cols[OIL_RATE_COL]]
    water_inj_rate = prod_df[prod_cols[WATER_INJ_COL]]
    gians = prod_df[prod_cols[RIG_COL]]
    marker_df = excel_utils.parse_marker(marker_dir)
    log_details = []

    for wIdx, well in enumerate(well_names):
        loai = "N/A"
        gian = "N/A"
        prod_row: int | None = next(
            iter(prod_df.index[prod_df[prod_cols[WELL_COL]] == well]), None
        )
        if prod_row is not None:
            if oil_rate[prod_row] > 0:
                loai = "Khai thác"
            elif water_inj_rate[prod_row] > 0:
                loai = "Bơm ép"
            gian = gians[prod_row]
        loai_gieng_result[wIdx] = loai
        ten_gian_result[wIdx] = gian
        las_dir = os.path.join(wells_dir, well, "GIS", "Las")
        las_file_paths = [
            f.path
            for f in os.scandir(las_dir)
            if f.is_file() and f.name.lower().endswith(".las")
        ]
        log_result[wIdx] = "yes" if len(las_file_paths) > 0 else ""
        for las_file_path in las_file_paths:
            try:
                res = json.loads(enhanced_las_parser(las_file_path)["text"])
                if res.get('error') is None:
                    log_details.append(res)
            except Exception as e:
                raise Exception(f"Error parsing las file {las_file_path}: {e}")
        devi_dir = os.path.join(wells_dir, well, "GIS", "Devi")
        devi_result[wIdx] = (
            "yes" if os.path.exists(devi_dir) and os.scandir(devi_dir) else ""
        )
        mudlog_dir = os.path.join(wells_dir, well, "GIS", "Master logs")
        mudlog_result[wIdx] = (
            "yes"
            if os.path.exists(mudlog_dir)
            and any(
                f.name.lower().endswith((".asc", ".pdf"))
                for f in os.scandir(mudlog_dir)
            )
            else ""
        )
        kqdvl_dir = os.path.join(wells_dir, well, "GIS", "Bao cao DVL")
        kqdvl_result[wIdx] = (
            "yes" if os.path.exists(kqdvl_dir) and os.scandir(kqdvl_dir) else ""
        )
        marker_result[wIdx] = (
            "yes" if (marker_df[marker_df.columns[0]] == well).any() else ""
        )

    return (
        well_names,
        loai_gieng_result,
        ten_gian_result,
        log_result,
        devi_result,
        mudlog_result,
        marker_result,
        thu_via_result,
        plt_result,
        kqdvl_result,
        log_details,
    )
