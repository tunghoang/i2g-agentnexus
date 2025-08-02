import os
import json
import numpy as np
import lasio
from datetime import datetime

import pandas as pd
from naming import Naming
from robust_las_parser import load_las_file
from xlsx_utils import XLSX


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

    marker_df = XLSX.parse_marker()

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


def read_curves_from_las(well_name: str, curves: list[str], wells_dir: str) -> np.ndarray | None:
    las_dir = os.path.join(wells_dir, well_name, "GIS", "Las")
    las_files = [
        f.path for f in os.scandir(las_dir)
        if f.is_file() and f.name.lower().endswith(".las")
    ]
    if not las_files:
        raise FileNotFoundError(f"No las files found")
    
    las_file_path = las_files[0]
    las, error = load_las_file(las_file_path)
    if las is None:
        raise Exception(f"Error parsing las file {las_file_path}: {error}")
    
    data = []
    for curve in curves:
        curve_data = las.get_robust_curve_data(curve)
        if len(curve_data) > 0:
            data.append(curve_data)

    if len(data) != len(curves):
        return None

    stacked = np.vstack(data).T
    valid_mask = ~np.isnan(stacked).any(axis=1)
    return stacked[valid_mask]

def write_curve_to_las(
        well_name: str, 
        curves: list[str], 
        curve_name: str, 
        curve_data: np.ndarray, 
        wells_dir: str
    ) -> bool:

    las_dir = os.path.join(wells_dir, well_name, "GIS", "Las")
    las_files = [
        f.path for f in os.scandir(las_dir)
        if f.is_file() and f.name.lower().endswith(".las")
    ]
    if not las_files:
        raise FileNotFoundError(f"No las files found")

    las_file_path = las_files[0]
    las, error = load_las_file(las_file_path)
    if las is None:
        raise Exception(f"Error parsing las file {las_file_path}: {error}")

    data = []
    for curve in curves:
        curve_data = las.get_robust_curve_data(curve)
        if len(curve_data) > 0:
            data.append(curve_data)

    if len(data) != len(curves) or len(data) == 0:
        return False

    stacked = np.vstack(data).T
    masks = ~np.isnan(stacked).any(axis=1)
    full_curve_data = np.full(len(masks), np.nan)
    j = 0

    for (i, mask) in enumerate(masks):
        if j >= len(curve_data):
            break
        if not mask:
            continue
        full_curve_data[i] = curve_data[j]
        j += 1

    if not las.las_obj:
        return False

    las_obj = las.las_obj
    
    if las.curve_exists(curve_name):
        las_obj[curve_name] = full_curve_data
    else:
        las_obj.curves.append(lasio.CurveItem(mnemonic=curve_name, data=full_curve_data, descr="Predicted curve"))

    output_path = os.path.join(las_dir, "pseudo_log", f"{curve_name}.las")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    las_obj.write(output_path)
    return True

def prepare_las_training_data(wells: list[str], curves: list[str], wells_dir: str) -> np.ndarray:
    all_data = []
    for well in wells:
        data = read_curves_from_las(well, curves, wells_dir)
        if data is not None:
            all_data.append(data)

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
    
def human_readable_diff(start_time: datetime, end_time: datetime) -> str:
    if not start_time or not end_time:
        return ""

    seconds_total = int((end_time - start_time).total_seconds())
    minutes, seconds = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")

    return ' '.join(parts)

def make_pseudo_log(
        target_curve: str = '',
        target_well: str = '',
        input_curves: list[str] = [],
        wells: list[str] = [],
        regression_model: str = "random_forest",
        params: dict = {},
        mlflow_uri: str = "http://localhost:5000",
        wells_dir: str = "data/wells",
    ):

    available_wells = [entry.name for entry in os.scandir(wells_dir) if entry.is_dir()]
    selected_wells = [name for name in available_wells if name in wells] if wells else available_wells
    selected_wells.sort()

    if len(selected_wells) == 0:
        raise Exception(f"No wells {wells} found")
    
    if target_well not in available_wells:
        raise Exception(f"No well {target_well} found")
    
    if len(input_curves) == 0:
        raise Exception(f"No curves {input_curves} found")
    
    import mlflow
    model_name = f"{target_curve}_{target_well}_{regression_model}"
    mlflow.set_tracking_uri(mlflow_uri)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("target_curve", target_curve)
        mlflow.log_param("regression_model", regression_model)
        mlflow.log_param("input_curves", input_curves)
        mlflow.log_param("input_wells", wells)
        mlflow.log_param("model_params", params)

        curves = [c for c in input_curves if c != target_curve] + [target_curve]
        data = prepare_las_training_data(wells, curves, wells_dir) 
        if len(data) == 0:
            raise ValueError(f"No valid data found for {curves} in wells {wells}")
        
        if len(data) < 10:
            raise ValueError(f"Insufficient data to train a model: only {len(data)} samples found")
        
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
        
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        model = train_model(x_train, y_train, regression_model, **params)
        mlflow.sklearn.log_model(model, name=model_name, input_example=x_train[:5])

        # log metrics
        from sklearn.metrics import (
            mean_squared_error, r2_score,
            mean_absolute_percentage_error,
            mean_absolute_error, max_error,
            explained_variance_score,
        )
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        y_pred = model.predict(x_test)

        mask = y_test != 0
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) * 100
        max_err = max_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        ev = explained_variance_score(y_test, y_pred)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("max_error", max_err)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("explained_variance", ev)

        # write to las file
        write_success = False
        target_input = prepare_las_training_data([target_well], input_curves, wells_dir)
        if len(target_input) > 0:
            predicted_curve = model.predict(target_input)
            write_success = write_curve_to_las(target_well, input_curves, target_curve, predicted_curve, wells_dir)
        mlflow.log_metric("saved_las_file", int(write_success))
        
        # evaluation
        fig = visualize_training_result(model, data)
        out_html = os.path.join("/tmp", "plot.html")
        fig.write_html(out_html)
        mlflow.log_artifact(out_html, artifact_path="plots")

def visualize_training_result(model, data: np.ndarray):
    from sklearn.model_selection import learning_curve
    import plotly.graph_objects as go

    x, y = data[:, :-1], data[:, -1]
    train_sizes, train_scores, val_scores = learning_curve(
        model, x, y,
        cv=3,
        scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        error_y=dict(type='data', array=train_std),
        mode='lines+markers', name='Training Score'
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        error_y=dict(type='data', array=val_std),
        mode='lines+markers', name='Validation Score'
    ))
    fig.update_layout(
        title='Learning Curve',
        xaxis_title='Training Set Size',
        yaxis_title='R² Score',
        template='plotly_white'
    )
    return fig


def get_training_result(
        target_curve: str = '',
        target_well: str = '',
        regression_model: str = "random_forest",
        mlflow_uri: str = "http://localhost:5000",
    ):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    if not experiment:
        raise ValueError("MLflow experiment 'Default' not found.")

    model_name = f"{target_curve}_{target_well}_{regression_model}"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], 
        filter_string=f"tags.mlflow.runName='{model_name}'",
        order_by=["start_time DESC"], 
        max_results=5
    )
    model_names: list[str] = []
    time_status: list[str] = []
    status_list: list[str] = []
    durations: list[str] = []
    details: list[str] = []

    template_path = "templates/training_result_report_tpl.html"
    with open(template_path, "r", encoding="utf-8") as tpl_file:
        template = tpl_file.read()

    for idx, run in enumerate(runs):
        out_file_path = f"{idx}_training_result_report.html"
        data = run.data
        run_name = run.info.run_name or "N/A"
        run_status = run.info.status
        
        start_time = datetime.fromtimestamp(run.info.start_time / 1000.0)
        end_time = datetime.fromtimestamp(run.info.end_time / 1000.0) if run.info.end_time else None
        now = datetime.now()
        time_since_run = human_readable_diff(start_time, now)
        time_since_run = time_since_run.split(" ")[0] + " ago" if time_since_run else "N/A"
        run_duration = human_readable_diff(start_time, end_time)
        run_duration = run_duration.split(" ")[0] if run_duration else "N/A"
        
        curve = data.params.get("target_curve", "N/A")
        input_curves = data.params.get("input_curves", [])
        input_wells = data.params.get("input_wells", [])
        model_params = data.params.get("model_params", {})
        model_type = data.params.get("regression_model", "N/A")
        mape_value = data.metrics.get("mape")
        mape = f"{mape_value:.2f}" if mape_value is not None else "N/A"
        rmse_value = data.metrics.get("rmse")
        rmse = f"{rmse_value:.4f}" if rmse_value is not None else "N/A"
        r2_value = data.metrics.get("r2")
        r2 = f"{r2_value:.4f}" if r2_value is not None else "N/A"
        dashboard_uri = mlflow_uri.replace("localhost", "dashboard.portal")
        dashboard = f'<a href="{dashboard_uri}" target="_blank">View on MLflow</a>'
        las_saved = "yes" if data.metrics.get("saved_las_file") == 1 else "no"

        df = pd.DataFrame([{
            "Model Name": run_name,
            "Target Curve": curve,
            "From Curves": input_curves,
            "From Wells": input_wells,
            "Model Type": model_type,
            "Params": model_params,
            "MAPE (%)": mape,
            "RMSE": rmse,
            "R² Score": r2,
            "Evaluation Dashboard": dashboard,
            "Las File Saved": las_saved
        }])
        table = df.to_html(index=False, escape=False)
        plot_path = f"mlartifacts/0/{run.info.run_id}/artifacts/plots/plot.html"
        if os.path.isfile(plot_path):
            with open(plot_path, "r") as f:
                plot = f.read()
        else:
            plot = "Evaluating..."

        result = template.replace("{{TABLE}}", table).replace("{{PLOT}}", plot)
        with open(os.path.join("/tmp", out_file_path), "w") as output_file:
            output_file.write(result)

        model_names.append(run_name)
        time_status.append(time_since_run)
        status_list.append(run_status)
        durations.append(run_duration)
        details.append(
            f'<a href="{out_file_path}" target="_blank">view</a>' #if run_status == "FINISHED" else "N/A"
        )

    return (
        model_names,
        time_status,
        status_list,
        durations,
        details
    )
