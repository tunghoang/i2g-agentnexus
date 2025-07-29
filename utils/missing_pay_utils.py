import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import utils.excel_utils as excel_utils
from naming import Naming
from robust_las_parser import load_las_file


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
    marker_df = excel_utils.parse_marker(marker_path)

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
    lld_result: list[str] = [""] * count
    bk_result: list[str] = [""] * count
    resdt_result: list[str] = [""] * count
    ild_result: list[str] = [""] * count
    rt_result: list[str] = [""] * count
    llm_result: list[str] = [""] * count
    lls_result: list[str] = [""] * count
    msfl_result: list[str] = [""] * count
    rxo_result: list[str] = [""] * count
    rhob_result: list[str] = [""] * count
    nphi_result: list[str] = [""] * count
    dt_result: list[str] = [""] * count
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
                if any(c in ["CAL", "CALI", "CALIPER"] for c in curve_names):
                    cal_result[wIdx] = "yes"
                if "LLD" in curve_names:
                    lld_result[wIdx] = "yes"
                if "BK" in curve_names:
                    bk_result[wIdx] = "yes"
                if "RESDT" in curve_names:
                    resdt_result[wIdx] = "yes"
                if "ILD" in curve_names:
                    ild_result[wIdx] = "yes"
                if "RT" in curve_names:
                    rt_result[wIdx] = "yes"
                if "LLM" in curve_names:
                    llm_result[wIdx] = "yes"
                if "LLS" in curve_names:
                    lls_result[wIdx] = "yes"
                if "MSFL" in curve_names:
                    msfl_result[wIdx] = "yes"
                if "RXO" in curve_names:
                    rxo_result[wIdx] = "yes"
                if any(c in ["RHOB", "RBOB"] for c in curve_names):
                    rhob_result[wIdx] = "yes"
                if "NPHI" in curve_names:
                    nphi_result[wIdx] = "yes"
                if "DT" in curve_names:
                    dt_result[wIdx] = "yes"
                if "PE" in curve_names:
                    pe_result[wIdx] = "yes"
            except Exception as e:
                raise Exception(f"Error parsing las file {las_file_path}: {e}")

    return (
        well_names,
        gr_result,
        sp_result,
        cal_result,
        lld_result,
        bk_result,
        resdt_result,
        ild_result,
        rt_result,
        llm_result,
        lls_result,
        msfl_result,
        rxo_result,
        rhob_result,
        nphi_result,
        dt_result,
        pe_result,
    )


def read_curves_from_las(las_file_path: str, curve_types: list[str]):
    las, error = load_las_file(las_file_path)
    if las is None:
        raise Exception(f"Error parsing las file {las_file_path}: {error}")
    curve_names = las.get_curve_names()
    data = []
    for curve in curve_types:
        matched_curve = None
        for las_curve in curve_names:
            if curve in str.upper(las_curve):
                matched_curve = las_curve
                break
        if not matched_curve:
            return None
        data.append(las.get_curve_data(matched_curve))

    stacked = np.vstack(data).T
    if np.any(np.isnan(stacked)):
        stacked = np.nan_to_num(stacked, nan=0.0)
    return stacked


def load_las_data(wells: list[str], curve_types: list[str], wells_dir: str):
    all_data = []
    for well in wells:
        las_dir = os.path.join(wells_dir, well, "GIS", "Las")
        las_file_paths = [
            f.path
            for f in os.scandir(las_dir)
            if f.is_file() and f.name.lower().endswith(".las")
        ]
        for las_file_path in las_file_paths:
            data = read_curves_from_las(las_file_path, curve_types)
            if data is not None:
                all_data.append(data)
                break # match the first las file
    
    if not all_data:
        raise ValueError("No valid training data found.")
    return np.vstack(all_data)


def train_model(x_train, y_train, regression_model: str, **kwargs):
    models = {
        "random_forest": RandomForestRegressor,
        "linear_regression": LinearRegression
    }

    if regression_model not in models:
        raise ValueError(f"Unsupported model: '{regression_model}'")

    default_params = {
        "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "linear_regression": {}
    }

    model_params = {**default_params.get(regression_model, {}), **kwargs}
    model = models[regression_model](**model_params)
    model.fit(x_train, y_train)
    return model


def generate_curve(
        target_curve: str, 
        target_well: str, 
        input_curves: list[str], 
        training_wells: list[str], 
        regression_model: str, 
        params: dict,
        wells_dir: str = "data/wells",
    ):
    model = None
    if len(training_wells) > 10:
        # get pretrained model
        model_file = Naming.model_file(target_curve)
        if os.path.exists(model_file):
            model = joblib.load(model_file)

    if not model:
        training_data = load_las_data(training_wells, list(set(input_curves + [target_curve])), wells_dir)
        x_train = training_data[:, :-1]
        y_train = training_data[:, -1]
        model = train_model(x_train, y_train, regression_model, **params)

    test_data = load_las_data([target_well], input_curves, wells_dir)
    if test_data is None or len(test_data) == 0:
        return

    y_pred = model.predict(test_data)
    return y_pred


def make_psuedo_log(
        psuedo_log: str = '',
        well: str = '',
        logs: list[str] = [],
        wells: list[str] = [],
        regression_model: str = "random_forest",
        params: dict = {},
        wells_dir: str = "data/wells",
    ):
    available_wells = [entry.name for entry in os.scandir(wells_dir) if entry.is_dir()]

    if wells:
        well_names = [name for name in available_wells if name in wells]
    else:
        well_names = available_wells
    
    well_names.sort()
    if len(well_names) == 0:
        raise Exception(f"No wells found for {wells}")
    
    if well not in available_wells:
        raise Exception(f"No well {well} found")
    
    if len(logs) == 0:
        raise Exception(f"No logs found")
    
    return generate_curve(psuedo_log, well, logs, wells, regression_model, params, wells_dir)

