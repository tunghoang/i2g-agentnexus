from datetime import datetime
import os
import pandas as pd
from naming import Naming
from robust_las_parser import load_las_file
import utils.excel_utils as excel_utils


def get_well_checklist(
    wells: list[str] = [],
    wells_dir: str = "data/wells",
    marker_path: str = "data/misc/Marker.xlsx",
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
    kb_result: list[str] = ["N/A"] * count
    nam_khoan_result: list[str] = ["N/A"] * count
    mong_result: list[str] = ["N/A"] * count
    day_md_result: list[str] = ["N/A"] * count
    day_tvdss_result: list[str] = ["N/A"] * count
    doi_tuong_khoan_result: list[str] = ["N/A"] * count
    log_result: list[str] = ["N/A"] * count
    devi_result: list[str] = ["N/A"] * count
    mudlog_result: list[str] = ["N/A"] * count
    marker_result: list[str] = ["N/A"] * count
    thu_via_result: list[str] = ["N/A"] * count
    plt_result: list[str] = ["N/A"] * count
    kqdvl_result: list[str] = ["N/A"] * count

    elevation_path = Naming.elevation_file()
    ELEVATION_WELL_COL = 2
    elevation_df = pd.read_excel(elevation_path, header=1)
    elevation_well = elevation_df[elevation_df.columns[ELEVATION_WELL_COL]].astype(str)
    loai_giengs = elevation_df[elevation_df.columns[3]]
    gians = elevation_df[elevation_df.columns[4]]
    kbs = elevation_df[elevation_df.columns[5]]
    ngay_khoans = elevation_df[elevation_df.columns[6]]
    do_sau_mongs = elevation_df[elevation_df.columns[8]]
    day_mds = elevation_df[elevation_df.columns[10]]
    day_tvdss = elevation_df[elevation_df.columns[11]]
    doi_tuong_khoans = elevation_df[elevation_df.columns[12]]

    marker_df = excel_utils.parse_marker(marker_path)

    for wIdx, well in enumerate(well_names):
        elevation_row: int | None = next(
            iter(elevation_well.index[elevation_well == well]), None
        )
        loai_gieng_result[wIdx] = loai_giengs.get(elevation_row) or "N/A"
        ten_gian_result[wIdx] = gians.get(elevation_row) or "N/A"
        kb_result[wIdx] = kbs.get(elevation_row) or "N/A"
        date_str = ngay_khoans.get(elevation_row)
        nam_khoan_result[wIdx] = (
            datetime.strptime(date_str, "%d.%m.%Y").year.__str__()
            if date_str
            else "N/A"
        )
        do_sau_mong: int | str | None = do_sau_mongs.get(elevation_row)
        mong_result[wIdx] = (
            "yes"
            if do_sau_mong is not None
            and (isinstance(do_sau_mong, str) and do_sau_mong.isdigit())
            or not isinstance(do_sau_mong, str)
            else ""
        )
        day_md_result[wIdx] = day_mds.get(elevation_row) or "N/A"
        day_tvdss_result[wIdx] = day_tvdss.get(elevation_row) or "N/A"
        doi_tuong_khoan_result[wIdx] = doi_tuong_khoans.get(elevation_row) or "N/A"

        las_dir = os.path.join(wells_dir, well, "GIS", "Las")
        log_result[wIdx] = (
            "yes" if os.path.exists(las_dir) and os.scandir(las_dir) else ""
        )
        devi_dir = Naming.devi_path(well)
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
        kb_result,
        nam_khoan_result,
        mong_result,
        day_md_result,
        day_tvdss_result,
        doi_tuong_khoan_result,
        log_result,
        devi_result,
        mudlog_result,
        marker_result,
        thu_via_result,
        plt_result,
        kqdvl_result,
    )


def get_well_checklist_curves(
    wells: list[str] = [],
    wells_dir: str = "data/wells",
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

    gr_result: list[str] = [""] * count
    sp_result: list[str] = [""] * count
    cal_result: list[str] = [""] * count
    deep_res_result: list[str] = [""] * count
    med_res_result: list[str] = [""] * count
    shal_res_result: list[str] = [""] * count
    micro_res_result: list[str] = [""] * count
    density_result: list[str] = [""] * count
    neutron_result: list[str] = [""] * count
    sonic_result: list[str] = [""] * count
    pe_result: list[str] = [""] * count

    for wIdx, well in enumerate(well_names):
        las_dir = os.path.join(wells_dir, well, "GIS", "Las")
        las_file_paths = [
            f.path
            for f in os.scandir(las_dir)
            if f.is_file() and f.name.lower().endswith(".las")
        ]
        for las_file_path in las_file_paths:
            try:
                las, error = load_las_file(las_file_path)
                if las is None:
                    raise Exception(f"Error parsing las file {las_file_path}: {error}")
                curve_names = [str.upper(c) for c in las.get_curve_names()]
                if "GR" in curve_names:
                    gr_result[wIdx] = "yes"
                if "SP" in curve_names:
                    sp_result[wIdx] = "yes"
                cal_curves = [
                    c for c in curve_names if c in ["CAL", "CALI", "CALIPER", "UCAV"]
                ]
                if len(cal_curves) > 0:
                    cal_result[wIdx] = f"yes - {', '.join(cal_curves)}"
                deep_res_curves = [
                    c
                    for c in curve_names
                    if c in ["LLD", "BK", "RESDT", "ILD", "RT", "P40H"]
                ]
                if len(deep_res_curves) > 0:
                    deep_res_result[wIdx] = f"yes - {', '.join(deep_res_curves)}"
                med_res_curves = [c for c in curve_names if c in ["P22H", "P34H"]]
                if len(med_res_curves) > 0:
                    med_res_result[wIdx] = f"yes - {', '.join(med_res_curves)}"
                shal_res_curves = [c for c in curve_names if c in ["LLS", "P16H"]]
                if len(shal_res_curves) > 0:
                    shal_res_result[wIdx] = f"yes - {', '.join(shal_res_curves)}"
                micro_res_curves = [c for c in curve_names if c in ["MSFL", "RXO"]]
                if len(micro_res_curves) > 0:
                    micro_res_result[wIdx] = f"yes - {', '.join(micro_res_curves)}"
                density_curves = [
                    c for c in curve_names if c in ["RHOB", "RBOB", "ROBB"]
                ]
                if len(density_curves) > 0:
                    density_result[wIdx] = f"yes - {', '.join(density_curves)}"
                neutron_curves = [c for c in curve_names if c in ["NPHI", "TNPH"]]
                if len(neutron_curves) > 0:
                    neutron_result[wIdx] = f"yes - {', '.join(neutron_curves)}"
                if "DT" in curve_names:
                    sonic_result[wIdx] = "yes"
                if "PE" in curve_names:
                    pe_result[wIdx] = "yes"
            except Exception as e:
                raise Exception(f"Error parsing las file {las_file_path}: {e}")

    return (
        well_names,
        gr_result,
        sp_result,
        cal_result,
        deep_res_result,
        med_res_result,
        shal_res_result,
        micro_res_result,
        density_result,
        neutron_result,
        sonic_result,
        pe_result,
    )
