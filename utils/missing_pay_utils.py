import os
import numpy as np
from datetime import datetime

import utils.excel_utils as excel_utils
import pandas as pd
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


def read_curves_from_las(las_file_path: str, curves: list[str]) -> np.ndarray:
    las, error = load_las_file(las_file_path)
    if las is None:
        raise Exception(f"Error parsing las file {las_file_path}: {error}")
    curve_names = las.get_curve_names()
    data = []
    for curve in curves:
        matched_curve = next(
            (las_curve for las_curve in curve_names if curve.upper() in las_curve.upper()),
            None
        )
        if not matched_curve:
            return None
        data.append(las.get_curve_data(matched_curve))

    stacked = np.vstack(data).T
    if np.any(np.isnan(stacked)):
        stacked = np.nan_to_num(stacked, nan=0.0)
    return stacked

def load_las_data(wells: list[str], curves: list[str], wells_dir: str) -> np.ndarray:
    all_data = []
    for well in wells:
        las_dir = os.path.join(wells_dir, well, "GIS", "Las")
        las_file_paths = [
            f.path
            for f in os.scandir(las_dir)
            if f.is_file() and f.name.lower().endswith(".las")
        ]
        for las_file_path in las_file_paths:
            data = read_curves_from_las(las_file_path, curves)
            if data is not None:
                all_data.append(data)
                break # match the first las file

    return np.vstack(all_data) if all_data else np.empty(0)

def train_model(x_train: np.ndarray, y_train: np.ndarray, regression_model: str, **kwargs):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor

    models = {
        "decision_tree": DecisionTreeRegressor,
        "huber": HuberRegressor,
        "lasso": Lasso,
        "linear": LinearRegression,
        "neural_network": MLPRegressor,
        "random_forest": RandomForestRegressor,
        "svm": SVR,
        "xgboost": XGBRegressor,
    }

    if regression_model not in models:
        raise ValueError(f"Unsupported model: '{regression_model}'")

    default_params = {
        "decision_tree": {},
        "huber": {},
        "lasso": {},
        "linear": {},
        "neural_network": {"hidden_layer_sizes": (100,), "activation": "relu", "max_iter": 500},
        "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "svm": {},
        "xgboost": {},
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
    ) -> np.ndarray:
    import joblib
    model = None
    if len(training_wells) > 10:
        # get pretrained model
        model_path = Naming.model_file(target_curve)
        if os.path.exists(model_path):
            model = joblib.load(model_path)

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
    
def generate_model(
    target_curve: str,
    target_well: str,
    input_curves: list[str],
    wells: list[str],
    regression_model: str,
    params: dict,
    wells_dir: str,
):
    import mlflow
    model_name = f"{target_well}_{target_curve}_{regression_model}"
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("target_curve", target_curve)
        mlflow.log_param("regression_model", regression_model)
        mlflow.log_param("input_curves", input_curves)
        mlflow.log_param("input_wells", wells)
        mlflow.log_params(params)

        curves = [c for c in input_curves if c != target_curve] + [target_curve]
        data = load_las_data(wells, curves, wells_dir) 
        if data.size == 0:
            raise ValueError(f"No valid data found for {curves} in wells {wells}")
        
        split_index = int(len(wells) * 0.8)
        training_data = data[:split_index]
        testing_data = data[split_index:]
        if training_data.size == 0 or testing_data.size == 0:
            raise ValueError(f"Insufficient data for training")

        #try:
        #    model_uri = f"models:/{model_name}/latest"
        #    model = mlflow.sklearn.load_model(model_uri)
        #except Exception:
        x_train = training_data[:, :-1]
        y_train = training_data[:, -1]
        model = train_model(x_train, y_train, regression_model, **params)
        mlflow.sklearn.log_model(model, name=model_name, input_example=x_train[:1])

        # log metrics
        from sklearn.metrics import mean_squared_error, r2_score
        x_test = testing_data[:, :-1]
        y_test = testing_data[:, -1]
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2_score(y_test, y_pred))
        
        # return prediction for target curve
        test_data = load_las_data([target_well], input_curves, wells_dir)
        if test_data is None or len(test_data) == 0:
            return
        return model.predict(test_data)

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
    well_names = [name for name in available_wells if name in wells] if wells else available_wells
    well_names.sort()

    if len(well_names) == 0:
        raise Exception(f"No wells found for {wells}")
    
    if well not in available_wells:
        raise Exception(f"No well {well} found")
    
    if len(logs) == 0:
        raise Exception(f"No logs found")
    
    return generate_model(psuedo_log, well, logs, well_names, regression_model, params, wells_dir)


