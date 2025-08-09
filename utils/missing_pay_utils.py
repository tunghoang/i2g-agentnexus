import os
import json
import uuid
import lasio
import numpy as np
import pandas as pd

from multiprocessing import Event
from datetime import datetime, timedelta, timezone
from naming import Naming
from robust_las_parser import load_las_file
from xlsx_utils import XLSX

import mlflow
from mlflow.tracking import MlflowClient
mlflow_uri = "http://localhost:5000"
mlflow.set_tracking_uri(mlflow_uri)


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


def read_curves_from_las(well_name: str, curves: list[str]) -> np.ndarray | None:
    las_dir = Naming.las_path(well_name)
    las_file_paths = [
        f.path for f in os.scandir(las_dir)
        if f.is_file() and f.name.lower().endswith(".las")
    ]
    if not las_file_paths:
        raise FileNotFoundError(f"No las files found")
    
    las_file_path = las_file_paths[0]
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
    ) -> str | None:

    las_dir = Naming.las_path(well_name)
    las_file_paths = [
        f.path for f in os.scandir(las_dir)
        if f.is_file() and f.name.lower().endswith(".las")
    ]
    if not las_file_paths:
        raise FileNotFoundError(f"No las files found")

    las_file_path = las_file_paths[0]
    las, error = load_las_file(las_file_path)
    if las is None:
        raise Exception(f"Error parsing las file {las_file_path}: {error}")

    data = []
    for curve in curves:
        curve_data = las.get_robust_curve_data(curve)
        if len(curve_data) > 0:
            data.append(curve_data)

    if len(data) != len(curves) or len(data) == 0:
        return None

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
        return None

    las_obj = las.las_obj
    
    if las.curve_exists(curve_name):
        las_obj[curve_name] = full_curve_data
    else:
        las_obj.curves.append(lasio.CurveItem(mnemonic=curve_name, data=full_curve_data, descr="Predicted curve"))

    las_file = f"{uuid.uuid4().hex[:8]}.las"
    las_obj.write(las_file)
    return las_file 

def prepare_las_training_data(wells: list[str], curves: list[str]) -> np.ndarray:
    all_data = []
    for well in wells:
        data = read_curves_from_las(well, curves)
        if data is not None:
            all_data.append(data)

    return np.vstack(all_data) if all_data else np.empty(0)

def train_model(x_train: np.ndarray, y_train: np.ndarray, model_type: str, **kwargs):
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

    if model_type not in models:
        raise ValueError(f"Unsupported model: '{model_type}'")

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

    model_params = {**default_params.get(model_type, {}), **kwargs}
    model = models[model_type](**model_params)
    model.fit(x_train, y_train)
    return model
    
def human_readable_diff(start_time: datetime, end_time: datetime) -> str:
    if not start_time or not end_time:
        return "N/A"

    delta = end_time - start_time
    if delta.days > 3:
        utc7 = timezone(timedelta(hours=7))
        return start_time.astimezone(utc7).strftime("%d/%m/%Y, %I:%M:%S %p")

    seconds_total = int(delta.total_seconds())
    minutes, seconds = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    if days:
        return f"{days} days ago"
    if hours:
        return f"{hours} hours ago"
    if minutes:
        return f"{minutes} minutes ago"
    
    return f"{seconds} seconds ago"

def make_pseudo_log(
        target_curve: str = '',
        target_well: str = '',
        curves: list[str] = [],
        wells: list[str] = [],
        model_type: str = "random_forest",
        model_params: dict = {},
        started_event: Event = None,
        exp_name: str = "pseudo_logs",
    ):
    
    client = MlflowClient()
    experiment = get_mlflow_experiment(client, name=exp_name)
    model_name = f"{target_curve}_{target_well}_{model_type}"

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_name):
        mlflow.log_param("target_curve", target_curve)
        mlflow.log_param("target_well", target_well)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("input_curves", json.dumps(curves))
        mlflow.log_param("input_wells", json.dumps(wells))
        mlflow.log_param("model_params", json.dumps(model_params))

        if started_event:
            started_event.set()

        all_curves = [c for c in curves if c != target_curve] + [target_curve]
        dataset = prepare_las_training_data(wells, all_curves) 
        if len(dataset) == 0:
            raise ValueError(f"No valid data found for curves {curves} in wells {wells}")
        
        # training
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        model = train_model(x_train, y_train, model_type, **model_params)
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
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("explained_variance", ev)

        # write to las file
        tmp_las = None
        input_data = prepare_las_training_data([target_well], curves)
        if len(input_data) > 0:
            predicted_curve_data = model.predict(input_data)
            tmp_las = write_curve_to_las(target_well, curves, target_curve, predicted_curve_data)
            mlflow.log_artifact(tmp_las, artifact_path="las")
            os.remove(tmp_las)
        mlflow.log_param("las_file", tmp_las)
    
        # evaluation
        fig = visualize_training_result(model, dataset)
        tmp_html = "plot.html"
        fig.write_html(tmp_html)
        mlflow.log_artifact(tmp_html, artifact_path="plots")
        os.remove(tmp_html)

def visualize_training_result(model, data: np.ndarray):
    from sklearn.model_selection import learning_curve
    import plotly.graph_objects as go

    x, y = data[:, :-1], data[:, -1]
    train_sizes, train_scores, val_scores = learning_curve(
        model, x, y,
        cv=3,
        scoring='r2',
        train_sizes=np.linspace(0.2, 1.0, 5),
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

def normalize_filter_expr(expr: str) -> str:
    import re
    alias_map = {
        "loss": "mape",
        "accuracy": "r2",
    }
    metric_fields = ["r2", "mape", "rmse"]

    for alias, actual in alias_map.items():
        expr = re.sub(rf"\b(?<!\.){alias}\b", actual, expr)

    for field in metric_fields:
        expr = re.sub(rf"\b(?<!\.){field}\b", f"metrics.{field}", expr)

    return expr

def seconds_ago_to_timestamp(seconds_ago: int) -> int:
    dt = datetime.now() - timedelta(seconds=seconds_ago)
    return int(dt.timestamp()) * 1000

def make_filter_params(
    target_curve: str, 
    target_well: str, 
    model_type: str, 
    seconds: int, 
    filter_expr: str, 
) -> str:
    
    filter_params = []
    
    if target_curve:
        filter_params.append(f"params.target_curve = '{target_curve}'")
    
    if target_well:
        filter_params.append(f"params.target_well = '{target_well}'")
    
    if model_type:
        filter_params.append(f"params.model_type = '{model_type}'")
    
    if seconds:
        filter_params.append(f"attributes.start_time >= {seconds_ago_to_timestamp(seconds)}")
    
    if filter_expr:
        filter_params.append(normalize_filter_expr(filter_expr))
    
    return " and ".join(filter_params)
        
def parse_json_param(param: str):
    if not param:
        return "N/A"
    try:
        result = json.loads(param)
        if type(result) == list:
            return ", ".join(result)
        elif type(result) == dict:
            return ", ".join(f"{k}: {v}" for k, v in result.items())
        return result
    except:
        return param

def parse_float_param(param: float | int):
    if param is None:
        return "N/A"
    return f"{param:.2f}"

def get_mlflow_experiment(client: MlflowClient, name: str):
    experiment = client.get_experiment_by_name(name)
    if not experiment:
        experiment_id = client.create_experiment(name)
        experiment = client.get_experiment(experiment_id)
    return experiment

def get_mlflow_artifact_path(
        experiment_id: str,
        run_id: str,
        artifact_path: str,
        allow_remote: bool = False,
    ) -> str | None:

    mlflow_path = Naming.data_path(f"{experiment_id}/{run_id}/artifacts", prefix="./mlartifacts")
    local_file_path = os.path.join(mlflow_path, artifact_path)

    if os.path.isfile(local_file_path):
        return local_file_path 

    # MLflow is running on a different host
    if allow_remote:
        try:
            os.makedirs(dest_path, exist_ok=True)
            from mlflow.artifacts import download_artifacts
            downloaded_path = download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=dest_path
            )
            return downloaded_path
        except Exception:
            return None

    return None

def get_training_result(
        target_curve: str = '',
        target_well: str = '',
        model_type: str = '',
        seconds: int = 0,
        filter_expr: str = '',
        exp_name="pseudo_logs",
    ):

    client = MlflowClient()
    experiment = get_mlflow_experiment(client, name=exp_name)

    filter_params = make_filter_params(target_curve, target_well, model_type, seconds, filter_expr)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], 
        filter_string=filter_params,
        order_by=["start_time DESC"], 
        max_results=10
    )
    run_ids: list[str] = []
    model_names: list[str] = []
    curve_list: list[str] = []
    well_list: list[str] = []
    mapes: list[str] = []
    rmses: list[str] = []
    r2_scores: list[str] = []
    time_status: list[str] = []
    status_list: list[str] = []
    durations: list[str] = []
    details: list[str] = []

    template_path = "templates/training_result_report_tpl.html"
    with open(template_path, "r") as tpl_file:
        template = tpl_file.read()

    for idx, run in enumerate(runs):
        data = run.data
        run_id = run.info.run_id
        run_name = run.info.run_name or "N/A"
        run_status = run.info.status
        
        start_time = datetime.fromtimestamp(run.info.start_time / 1000.0)
        time_since_run = human_readable_diff(start_time, datetime.now())
        
        curve = data.params.get("target_curve", "N/A")
        well = data.params.get("target_well", "N/A")
        model = data.params.get("model_type", "N/A")
        
        input_curves = parse_json_param(data.params.get("input_curves"))
        input_wells = parse_json_param(data.params.get("input_wells"))
        model_params = parse_json_param(data.params.get("model_params"))
        mape = parse_float_param(data.metrics.get("mape"))
        rmse = parse_float_param(data.metrics.get("rmse"))
        r2 = parse_float_param(data.metrics.get("r2_score"))
        las_file = data.params.get("las_file") 
        las_file_path = get_mlflow_artifact_path(experiment.experiment_id, run_id, artifact_path=f"las/{las_file}")
        las_file_link = f"<a href='{las_file_path}'>Download</a>" if las_file else "N/A"

        # generate table
        df = pd.DataFrame([{
            "ID": run_id[:8],
            "Model Name": run_name,
            "Target Curve": curve,
            "Target Well": well,
            "From Curves": input_curves,
            "From Wells": input_wells,
            "Model Type": model,
            "Params": model_params,
            "MAPE (%)": mape,
            "RMSE": rmse,
            "R² Score": r2,
            "Las File": las_file_link
        }])
        table = df.to_html(index=False, escape=False)
        
        # generate plot
        plot_path = get_mlflow_artifact_path(experiment.experiment_id, run_id, artifact_path="plots/plot.html")
        if plot_path:
            with open(plot_path, "r") as f:
                plot = f.read()
        else:
            plot = "Evaluating..."

        each_report_html = template.replace("{{TABLE}}", table).replace("{{PLOT}}", plot)
        each_report_file = f"{idx}_training_result_report.html"
        each_report_path = Naming.dest_path(each_report_file, format="")
        each_report_link = f'<a href="{each_report_file}" target="_blank">{run_id[:8]}</a>'
        with open(each_report_path, "w") as f:
            f.write(each_report_html)

        run_ids.append(each_report_link)
        model_names.append(run_name)
        curve_list.append(input_curves)
        well_list.append(input_wells)
        mapes.append(mape)
        rmses.append(rmse)
        r2_scores.append(r2)
        time_status.append(time_since_run)
        status_list.append(run_status)

    df = pd.DataFrame(data={
        "ID": run_ids,
        "Model Name": model_names,
        "From Curves": curve_list,
        "From Wells": well_list,
        "MAPE (%)": mapes,
        "RMSE": rmses,
        "R² Score": r2_scores,
        "Created": time_status,
        "Status": status_list,
    })
    table = df.to_html(index=False, escape=False)
    
    report_html = template.replace("{{TABLE}}", table).replace("{{PLOT}}", "")
    report_file = "training_result_report.html"
    report_path = Naming.dest_path(report_file, format="")
    with open(report_path, "w") as f:
        f.write(report_html)

    return report_file 

def remove_training_result(run_id_prefix: str, exp_name="pseudo_logs") -> bool:
    client = MlflowClient()
    experiment = get_mlflow_experiment(client, name=exp_name)

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    matched_runs = [run for run in runs if run.info.run_id.startswith(run_id_prefix)]

    for run in matched_runs:
        client.delete_run(run.info.run_id)
    
    return True
    
