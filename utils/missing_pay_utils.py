import os
import re
import json
import uuid
import lasio
import yaml
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from glob import glob, iglob
from multiprocessing import Event
from datetime import datetime, timedelta, timezone
from cache import MemoryCache
from naming import Naming
from robust_las_parser import load_las_file, load_las_file_1
from xlsx_utils import XLSX
from store import Store

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.artifacts import download_artifacts

# regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    # regression
    mean_squared_error, r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error, max_error,
    explained_variance_score,
    # classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    auc,
    confusion_matrix, 
    roc_curve, 
    precision_recall_curve
)

from utils.plot_utils import logplot
from base_utils import aliases_of_curve, standard_curve_name, find_similar_curves, getUnit, get_mlflow_experiment, human_readable_diff, parse_json_param, parse_float_param, get_mlflow_artifact_path, seconds_ago_to_timestamp, excel_link, mark_sample

mlflow_uri = "http://localhost:5000"
mlflow.set_tracking_uri(mlflow_uri)

def get_well_checklist(
    wells: list[str] = [],
    wells_dir: str = "data/wells",
    marker_path: str = "data/misc/Marker.xlsx",
):
    print("wellllllls:", wells)
    if not os.path.isdir(wells_dir):
        raise Exception(f"Directory {wells_dir} does not exist")

    well_names = [f.name for f in os.scandir(wells_dir) if f.is_dir()]
    if wells is not None and len(wells) > 0:
        well_names = [f for f in well_names if f in wells]
    well_names.sort()
    #if len(well_names) > 5:
    #    well_names = well_names[:5]
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

def _get_well_checklist_curves_1(
    wells: list[str] = [],
    wells_dir: str = "data/wells",
):
    if not os.path.isdir(wells_dir):
        raise Exception(f"Directory {wells_dir} does not exist")

    well_names = [f.name for f in os.scandir(wells_dir) if f.is_dir()]
    if wells is not None and len(wells) > 0:
        well_names = [f for f in well_names if f in wells]
    well_names.sort()
    #if len(well_names) > 5:
    #    well_names = well_names[:5]
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
            if f.is_file() and ( not os.path.islink(f.path) ) and f.name.lower().endswith(".las")
        ]
        las_file_paths.sort()
        for las_file_path in las_file_paths:
            try:
                key_path = Naming.to_raw_path(las_file_path)
                _las = MemoryCache.get_instance().get(key_path)
                if _las is None:
                    _las = lasio.read(las_file_path)
                    MemoryCache.get_instance().put(key_path, _las)
                las, error = load_las_file_1(_las, las_file_path)
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
    #if len(well_names) > 5:
    #    well_names = well_names[:5]
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
    vshale_result: list[str] = [""] * count
    phie_result: list[str] = [""] * count
    sw_result: list[str] = [""] * count

    for wIdx, well in enumerate(well_names):
        curves = read_curves_meta_data_from_las(well)
        curve_names = [str.upper(c.mnemonic) for c in curves]
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
        if len(find_similar_curves('VSHALE', curve_names)) > 0:
            vshale_result = 'yes'
        if len(find_similar_curves('PHIE', curve_names)) > 0:
            phie_result = 'yes'
        if len(find_similar_curves('SW', curve_names)) > 0:
            sw_result = 'yes'

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
        vshale_result,
        phie_result,
        sw_result
    )
def _read_curves_from_las_file(las_file_path, curves: list[str], use_latest=False):
    ori_file_path = Naming.to_raw_path(las_file_path)
    las = MemoryCache.get_instance().get(ori_file_path)
    if las is None:
        las = lasio.read(las_file_path)
        MemoryCache.get_instance().put(ori_file_path, las)

    #df = las.df().reset_index()
    df = las.df()
    all_cols = df.columns
    if len(curves) > 0:
        selected_curves = []
        for c in curves:
            if c.startswith('$'): # exact match
                for c1 in all_cols:
                    if c1[1:] == c :
                        selected_curves.append(c1)
                        break
            else:
                curve_found = None
                for c1 in all_cols:
                    if standard_curve_name(c1) == standard_curve_name(c):
                        curve_found = c1
                        if not use_latest:
                            break
                if curve_found:
                    selected_curves.append(curve_found)
        df = df[ selected_curves ]
        #df = df[ [all_cols[0]] + selected_curves ]
    #df = df.set_index(all_cols[0])
    print(las_file_path, df.columns)
    return df

def read_curves_meta_data_from_las(well_name: str):
    las_dir = Naming.las_path(well_name)
    las_file_paths = [
        f.path for f in os.scandir(las_dir)
        if f.is_file() and ( not os.path.islink(f.path) ) and f.name.lower().endswith(".las")
    ]
    las_file_paths.sort()
    if not las_file_paths:
        raise FileNotFoundError(f"No las files found")

    curves = lasio.las_items.SectionItems()
    first_file = True
    for las_file_path in las_file_paths:
        ori_file_path = Naming.to_raw_path(las_file_path)
        las = MemoryCache.get_instance().get(ori_file_path)
        if las is None:
            las = lasio.read(las_file_path)
            MemoryCache.get_instance().put(ori_file_path, las)
        first_curve = True
        for c in las.curves:
            if (not first_curve) or (first_curve and first_file):
                curves.append(c)
            first_curve = False
        first_file = False
    return curves

def __rename_dup_columns(df):
    columns = pd.Series(df.columns)
    dup_names = columns[columns.duplicated()].tolist()
    counters = { n: 0 for n in dup_names }
    new_columns = []
    for c in columns:
        if c in dup_names:
            cnt = counters[c] + 1
            counters[c] = cnt
            new_columns.append(f"{c}:{cnt}")
        else:
            new_columns.append(c)

    df.columns = new_columns
    return df
    
def read_curves_from_las(well_name: str, curves: list[str], use_latest=False) -> pd.DataFrame | None:
    las_dir = Naming.las_path(well_name)
    las_file_paths = [
        f.path for f in os.scandir(las_dir)
        if f.is_file() and ( not os.path.islink(f.path) ) and f.name.lower().endswith(".las")
    ]
    las_file_paths.sort()
    if not las_file_paths:
        raise FileNotFoundError(f"No las files found")
    dfs = []
    print("LAS_PATHS_ARRAY", las_file_paths)
    for las_file_path in las_file_paths:
        #las_file_path = las_file_paths[0]
        df = _read_curves_from_las_file(las_file_path, curves, use_latest=use_latest)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df = __rename_dup_columns(df)
    return df

def read_curves_from_las_(well_name: str, curves: list[str]) -> np.ndarray | None:
    las_dir = Naming.las_path(well_name)
    las_file_paths = [
        f.path for f in os.scandir(las_dir)
        if f.is_file() and ( not os.path.islink(f.path) ) and f.name.lower().endswith(".las")
    ]
    las_file_paths.sort()
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

def write_curve_to_las( well_name: str, curve_name: str, curves:list[str], curve_data: np.ndarray, run_id='' ) -> str | None:
    df = read_curves_from_las(well_name, curves)
    df_cleaned = df.dropna()
    df_cleaned[curve_name] = curve_data
    merge_df = pd.concat([df, df_cleaned[curve_name]], axis=1)

    las_dir_path = Naming.las_path(well_name)
    las_file_paths = glob(f"{las_dir_path}/*.las")
    las_file_path = las_file_paths[0]
    ori_file_path = Naming.to_raw_path(las_file_path)
    las = MemoryCache.get_instance().get(ori_file_path)
    if las is None:
        las = lasio.read(las_file_path)
        MemoryCache.get_instance().put(ori_file_path, las)

    new_las = lasio.LASFile()
    new_las.version = las.version
    new_las.well = las.well
    new_las.params = las.params
    new_las.other = las.other
    new_las.append_curve(las.curves[0].mnemonic, merge_df.index, unit=las.curves[0].unit)
    new_las.append_curve(curve_name, merge_df[curve_name], unit=getUnit(curve_name))
    out_las_path = (run_id if run_id else uuid.uuid4().hex[:8]) + '.las'
    new_las.write(out_las_path)
    return out_las_path

def write_curve_to_las_(
        well_name: str, 
        curves: list[str], 
        curve_name: str, 
        curve_data: np.ndarray, 
    ) -> str | None:

    las_dir = Naming.las_path(well_name)
    las_file_paths = [
        f.path for f in os.scandir(las_dir)
        if f.is_file() and ( not os.path.islink(f.path) ) and f.name.lower().endswith(".las")
    ]
    las_file_paths.sort()
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

def prepare_las_training_data(wells: list[str], curves: list[str], with_zone=False, with_index=False, use_first=False) -> np.ndarray:
    all_data = []
    for well in wells:
        df = read_curves_from_las(well, curves)
        all_curves = list(df.columns)
        selected_curves = []
        for c in curves:
            candidate = find_similar_curves(c, all_curves)
            print("candidates:", candidate, c, all_curves, curves, well)
            if len(candidate) == 0:
                raise Exception(f"Curve {c} is not found in well {well}")
            if use_first:
                selected_curves.append(candidate[0])
            else:
                selected_curves.append(candidate[-1])
        

        df = df[selected_curves]
        if with_zone:
            keyzone, zone=XLSX.extract_zones1(well)
            index_col = df.index.name or 'index'
            target_series = df.reset_index()[index_col]
            zone['DEPTH'] = zone['start'].apply(lambda x: target_series.iloc[(abs(target_series - x)).idxmin()])
            zone = zone.set_index('DEPTH')
            df['Zone'] = zone['Surface']
            df['Zone'] = df['Zone'].ffill()
        if with_index:
            df = df.reset_index()

        df_cleaned = df.dropna()
        data = df_cleaned.values
        if data is not None:
            print("cccc", df_cleaned.columns)
            all_data.append(data)

    return np.vstack(all_data) if all_data else np.empty(0)

def train_model(x_train: np.ndarray, y_train: np.ndarray, model_type: str, **kwargs):

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

def train_classifier(x_train: np.ndarray, y_train: np.ndarray, model_type: str, **kwargs):
    models = {
        "decision_tree": DecisionTreeClassifier,
        "logistic": LogisticRegression,
        "neural_network": MLPClassifier,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "xgboost": XGBClassifier,
    }

    if model_type not in models:
        raise ValueError(f"Unsupported model: '{model_type}'")

    default_params = {
        "decision_tree": {"max_depth": 10, "random_state": 42},
        "logistic": {"max_iter": 1000},
        "neural_network": {"hidden_layer_sizes": (100,), "activation": "relu", "max_iter": 500},
        "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "svm": {"probability": True},  # for ROC AUC
        "xgboost": {"use_label_encoder": False, "eval_metric": "logloss"},
    }

    model_params = {**default_params.get(model_type, {}), **kwargs}
    model = models[model_type](**model_params)
    model.fit(x_train, y_train)
    return model

def make_pseudo_zones(
        target_well: str = '',
        curves: list[str] = [],
        wells: list[str] = [],
        model_type: str = "random_forest",
        model_params: dict = {},
        started_event: Event = None,
        exp_name: str = "pseudo_logs",
        #exp_name: str = "pseudo_logs_classifier",
    ):
    client = MlflowClient()
    experiment = get_mlflow_experiment(client, name=exp_name)
    model_name = f"Zone_{target_well}_{model_type}"

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_name) as run:
        mlflow.log_param("target_curve", "Zone")
        mlflow.log_param("target_well", target_well)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("input_curves", json.dumps(curves))
        mlflow.log_param("input_wells", json.dumps(wells))
        mlflow.log_param("model_params", json.dumps(model_params))

        if started_event:
            started_event.set()

        #all_curves = [c for c in curves]
        all_curves = curves
        dataset = prepare_las_training_data(wells, all_curves, with_zone = True)
        if len(dataset) == 0:
            raise ValueError(f"No valid data found for curves {curves} in wells {wells}")

        print(all_curves)
        # split
        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1] # labeled data
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1] # labeled data

        # train
        model = train_classifier(x_train, y_train, model_type, **model_params)
        #mlflow.sklearn.log_model(model, name=model_name, input_example=x_train[:5])
        mlflow.sklearn.log_model(sk_model=model, artifact_path='models', input_example=x_train[:5])

        # predict
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None

        # metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0, average='micro')
        rec = recall_score(y_test, y_pred, zero_division=0, average='micro')
        f1 = f1_score(y_test, y_pred, zero_division=0, average='micro')
        auc = roc_auc_score(y_test, y_proba, average='micro', multi_class='ovr') if y_proba is not None else None

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        # write result to file and create comparison logplot
        input_df = read_curves_from_las(target_well, curves)
        selected_curves = [
            find_similar_curves(c, input_df.columns)[-1] for c in curves
        ]
        input_df = input_df[selected_curves].dropna()
        input_data = input_df.values

        if input_data.size > 0:
            predicted_curve = model.predict(input_data)
            df = read_curves_from_las(target_well, curves)
            df_cleaned = df.dropna()
            df_cleaned['Zone'] = predicted_curve
            print(df_cleaned['Zone'].drop_duplicates(keep='first'))
            tmp_xlsx = f"{uuid.uuid4().hex[:8]}.xlsx"
            XLSX.save_dataframe(df_cleaned['Zone'].drop_duplicates(keep='first').reset_index(), tmp_xlsx)
            mlflow.log_artifact(tmp_xlsx, artifact_path="xlsx")
            mlflow.log_param("zone_file", os.path.basename(tmp_xlsx))
            os.remove(tmp_xlsx)
            
            '''
            # Comparison logplot
            input_df[f"{target_curve}:*"] = predicted_curve
            plot_df = read_curves_from_las(target_well, [target_curve])
            plot_df[f"{target_curve}:*"] = input_df[f"{target_curve}:*"]
            las_curves = read_curves_meta_data_from_las(target_well)
            las_curves.append(lasio.las_items.CurveItem(mnemonic=f"{target_curve}:*", unit="v/v"))

            fig = logplot(plot_df,las_curves)
            tmp_plot = "visualization.html"
            fig.write_html(tmp_plot)
            mlflow.log_artifact(tmp_plot, artifact_path="plots")
            os.remove(tmp_plot)
            '''
        
        # evaluation
        fig = visualize_zone_result(model, dataset)
        tmp_html = "plot.html"
        fig.write_html(tmp_html)
        mlflow.log_artifact(tmp_html, artifact_path="plots")
        os.remove(tmp_html)

def make_pseudo_log_classifier(
        target_curve: str = '',
        target_well: str = '',
        curves: list[str] = [],
        wells: list[str] = [],
        model_type: str = "random_forest",
        model_params: dict = {},
        started_event: Event = None,
        exp_name: str = "pseudo_logs",
        #exp_name: str = "pseudo_logs_classifier",
    ):
    client = MlflowClient()
    experiment = get_mlflow_experiment(client, name=exp_name)
    model_name = f"{target_curve}_{target_well}_{model_type}"

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_name) as run:
        mlflow.log_param("target_curve", target_curve)
        mlflow.log_param("target_well", target_well)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("input_curves", json.dumps(curves))
        mlflow.log_param("input_wells", json.dumps(wells))
        mlflow.log_param("model_params", json.dumps(model_params))

        if started_event:
            started_event.set()

        all_curves = [c for c in curves if c != target_curve] + [target_curve] # make sure that target curve is at the end of the list
        dataset = prepare_las_training_data(wells, all_curves)
        if len(dataset) == 0:
            raise ValueError(f"No valid data found for curves {curves} in wells {wells}")

        print(all_curves)
        # split
        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
        x_train = train_data[:, :-1]
        y_train = (train_data[:, -1] > 0).astype(int) # binary
        x_test = test_data[:, :-1]
        y_test  = (test_data[:, -1] > 0).astype(int)

        # train
        model = train_classifier(x_train, y_train, model_type, **model_params)
        #mlflow.sklearn.log_model(model, name=model_name, input_example=x_train[:5])
        mlflow.sklearn.log_model(sk_model=model, artifact_path='models', input_example=x_train[:5])

        # predict
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

        # metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        # write to las file and comparison logplot
        input_df = read_curves_from_las(target_well, curves)
        selected_curves = [
            find_similar_curves(c, input_df.columns)[-1] for c in curves
        ]
        input_df = input_df[selected_curves].dropna()
        input_data = input_df.values

        if input_data.size > 0:
            predicted_curve = model.predict(input_data)
            tmp_las = write_curve_to_las(
                target_well, 
                target_curve, 
                curves, 
                predicted_curve, 
                run_id=run.info.run_id
            )
            mlflow.log_artifact(tmp_las, artifact_path="las")
            mlflow.log_param("las_file", os.path.basename(tmp_las))
            os.remove(tmp_las)

            # Comparison logplot
            input_df[f"{target_curve}:*"] = predicted_curve
            plot_df = read_curves_from_las(target_well, [target_curve])
            plot_df[f"{target_curve}:*"] = input_df[f"{target_curve}:*"]
            las_curves = read_curves_meta_data_from_las(target_well)
            las_curves.append(lasio.las_items.CurveItem(mnemonic=f"{target_curve}:*", unit="v/v"))

            fig = logplot(plot_df,las_curves)
            tmp_plot = "visualization.html"
            fig.write_html(tmp_plot)
            mlflow.log_artifact(tmp_plot, artifact_path="plots")
            os.remove(tmp_plot)
        
        # evaluation
        fig = visualize_classifier_result(model, dataset)
        tmp_html = "plot.html"
        fig.write_html(tmp_html)
        mlflow.log_artifact(tmp_html, artifact_path="plots")
        os.remove(tmp_html)

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

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_name) as run:
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
        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        model = train_model(x_train, y_train, model_type, **model_params)
        #mlflow.sklearn.log_model(model, name=model_name, input_example=x_train[:5])
        mlflow.sklearn.log_model(sk_model=model, artifact_path='models', input_example=x_train[:5])

        # log metrics
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

        # write to las file and comparison logplot
        tmp_las = None
        input_df = read_curves_from_las(target_well, curves)
        selected_curves = []
        for c in curves:
            all_curves = input_df.columns
            candidates = find_similar_curves(c, all_curves)
            selected_curves.append(candidates[-1])
        input_df = input_df[selected_curves].dropna()
        input_data = input_df.values

        if len(input_data) > 0:
            print(input_data.shape)
            predicted_curve_data = model.predict(input_data)
            print(predicted_curve_data.shape)
            tmp_las = write_curve_to_las(target_well, target_curve, curves, predicted_curve_data, run_id=run.info.run_id)
            mlflow.log_artifact(tmp_las, artifact_path="las")
            os.remove(tmp_las)
            mlflow.log_param("las_file", os.path.basename(tmp_las))

            # Comparison logplot
            input_df[f"{target_curve}:*"] = predicted_curve_data
            plot_df = read_curves_from_las(target_well, [target_curve])
            plot_df[f'{target_curve}:*'] = input_df[f"{target_curve}:*"]
            las_curves = read_curves_meta_data_from_las(target_well)
            las_curves.append(lasio.las_items.CurveItem(mnemonic=f'{target_curve}:*', unit="v/v"))
            print(plot_df)
            fig = logplot(plot_df,las_curves)
            tmp_plot = 'visualization.html'
            fig.write_html(tmp_plot)
            mlflow.log_artifact(tmp_plot, artifact_path='plots')
            os.remove(tmp_plot)

        # evaluation
        fig = visualize_training_result(model, dataset)
        tmp_html = "plot.html"
        fig.write_html(tmp_html)
        mlflow.log_artifact(tmp_html, artifact_path="plots")
        os.remove(tmp_html)

def visualize_training_result(model, data: np.ndarray):
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

def visualize_zone_result(model, dataset):
    X = dataset[:, :-1]
    y = dataset[:, -1] # string labels
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y) # convert strings -> integers
    class_names = le.classes_
    n_classes = len(class_names)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
    else:
        raise ValueError("Model must support probability outputs (predict_proba).")

    y_pred = np.argmax(y_proba, axis=1)

    # Confusion Matrix
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Confusion Matrix", "ROC Curves", "Precision-Recall Curves"),
        horizontal_spacing=0.15
    )

    # Confusion Matrix
    cm = confusion_matrix(y_encoded, y_pred, labels=range(n_classes))
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=[f"Pred {c}" for c in class_names],
            y=[f"True {c}" for c in class_names],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=False
        ),
        row=1, col=1
    )

    # ROC Curves (One-vs-Rest)
    if y_proba is not None and n_classes > 1:
        y_bin = label_binarize(y_encoded, classes=range(n_classes))
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{cls} (AUC={roc_auc:.2f})"),
                row=1, col=2
            )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")),
            row=1, col=2
        )
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)

    # Precision-Recall Curves
    if y_proba is not None and n_classes > 1:
        y_bin = label_binarize(y_encoded, classes=range(n_classes))
        for i, cls in enumerate(class_names):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            fig.add_trace(
                go.Scatter(x=rec, y=prec, mode="lines", name=f"{cls}"),
                row=1, col=3
            )
        fig.update_xaxes(title_text="Recall", row=1, col=3)
        fig.update_yaxes(title_text="Precision", row=1, col=3)

    fig.update_layout(
        title="Classification Evaluation",
        width=1600,
        height=600,
        legend=dict(x=1.05, y=1)
    )

    return fig

def visualize_classifier_result(model, dataset):
    x = dataset[:, :-1]
    y = (dataset[:, -1] > 0).astype(int) # binary
    #y = dataset[:, -1]

    y_pred = model.predict(x)
    y_proba = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else None

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        horizontal_spacing=0.15
    )

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=False
        ),
        row=1, col=1
    )

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y, y_proba)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve"),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")),
            row=1, col=2
        )
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)

    # Precision-Recall Curve
    if y_proba is not None:
        prec, rec, _ = precision_recall_curve(y, y_proba)
        fig.add_trace(
            go.Scatter(x=rec, y=prec, mode="lines", name="PR curve"),
            row=1, col=3
        )
        fig.update_xaxes(title_text="Recall", row=1, col=3)
        fig.update_yaxes(title_text="Precision", row=1, col=3)

    fig.update_layout(
        title="Classification Evaluation Results",
        width=1400,
        height=500,
        showlegend=False
    )

    return fig

def normalize_filter_expr(expr: str) -> str:
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
        

def get_training_result(
        target_curve: str = '',
        target_well: str = '',
        model_type: str = '',
        seconds: int = 0,
        filter_expr: str = '',
        exp_name="pseudo_logs",
        #exp_name="pseudo_logs_classifier",
    ):
    print("======",target_curve, target_well, model_type, exp_name)
    client = MlflowClient()
    experiment = get_mlflow_experiment(client, name=exp_name)

    filter_params = make_filter_params(target_curve, target_well, model_type, seconds, filter_expr)
    print(filter_params)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], 
        filter_string=filter_params,
        order_by=["start_time DESC"], 
        max_results=10
    )
    print("RUNS:", runs)
    run_ids: list[str] = []
    model_names: list[str] = []
    curve_list: list[str] = []
    well_list: list[str] = []
    mapes: list[str] = []
    rmses: list[str] = []
    r2_scores: list[str] = []

    accuracies: list[str] = []
    f1_scores: list[str] = []
    recalls: list[str] = []
    precisions: list[str] = []
    roc_aucs: list[str] = []

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

        accuracy = parse_float_param(data.metrics.get('accuracy'))
        f1_score = parse_float_param(data.metrics.get('f1_score'))
        recall = parse_float_param(data.metrics.get('recall'))
        precision = parse_float_param(data.metrics.get('precision'))
        roc_auc = parse_float_param(data.metrics.get('roc_auc'))

        download_link = 'N/A'
        visualization_link = "N/A1"
        try:
            las_file = data.params.get("las_file")
            if las_file:
                las_file = os.path.basename(las_file)
                las_file_path = get_mlflow_artifact_path(experiment.experiment_id, run_id, artifact_path=f"las/{las_file}")
                download_link = f"<a href='{las_file_path}'>Downloadii</a>"

                visualization_path = get_mlflow_artifact_path(experiment.experiment_id, run_id, artifact_path="plots/visualization.html")
                visualization_link = f"<a href='{visualization_path}'>View</a>" if visualization_path else "N/A2"
            else:
                zone_file = data.params.get('zone_file')
                if zone_file:
                    zone_file = os.path.basename(zone_file)
                    zone_file_path = get_mlflow_artifact_path(experiment.experiment_id, run_id, artifact_path=f'xlsx/{zone_file}')
                    download_link = f"<a href='{zone_file_path}'>Download33</a>"
                    visualization_link = excel_link(zone_file_path, label='View', syntax='html')
                else:
                    download_link = "N/A"
                    visualization_link = 'N/A3'
        except Exception as e:
            traceback.print_exc()
            pass

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
            "Download": download_link,
            "View": visualization_link
        }])
        if curve in ['Zone', 'RESTF', 'PAYF']:
            df = pd.DataFrame([{
                "ID": run_id[:8],
                "Model Name": run_name,
                "Target Curve": curve,
                "Target Well": well,
                "From Curves": input_curves,
                "From Wells": input_wells,
                "Model Type": model,
                "Params": model_params,
                "Accuracy": accuracy,
                "F1": f1_score,
                "Recall": recall,
                "Precision": precision,
                'ROC_AUC': roc_auc,
                "Download": download_link,
                "View": visualization_link
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

        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        recalls.append(recall)
        precisions.append(precision)
        roc_aucs.append(roc_auc)

        time_status.append(time_since_run)
        status_list.append(run_status)

    df = pd.DataFrame(data={
        "ID": run_ids,
        "Model Name": model_names,
        "From Curves": curve_list,
        "From Wells": well_list,
        #"MAPE (%)": mapes,
        #"RMSE": rmses,
        "R² Score": r2_scores,
        "ROC_AUC": roc_aucs,
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

def get_runs(run_id_prefix: str, exp_name="pseudo_logs"):
    client = MlflowClient()
    experiment = get_mlflow_experiment(client, name = exp_name)
    runs = client.search_runs( experiment_ids=[experiment.experiment_id])
    matched_runs = [run for run in runs if run.info.run_id.startswith(run_id_prefix)]
    return matched_runs

def remove_training_result(run_id_prefix: str, exp_name="pseudo_logs") -> bool:
    #client = MlflowClient()
    #experiment = get_mlflow_experiment(client, name=exp_name)

    #runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    #matched_runs = [run for run in runs if run.info.run_id.startswith(run_id_prefix)]
    matched_runs = get_runs(run_id_prefix, exp_name=exp_name)
    print(matched_runs)
    for run in matched_runs:
        client = MlflowClient()
        client.delete_run(run.info.run_id)
    
    return True
    
def get_curves_in_well(well:str):
    storage = Store()
    curves = storage.get_curves_in_well(well)
    if curves is None:
        las_files = Naming.las_path(well) + "/*.las"
        all_curves = []
        for f in glob(las_files):
            if os.path.islink(f):
                continue
            las_f = lasio.read(f)
            curves = [{'curve': c.mnemonic, 'path': f, 'ref': None} for c in las_f.curves]
            all_curves += curves
        if len(all_curves):
            storage = Store()
            storage.save_curves_in_well(all_curves, well)
        return list(set([c['curve'] for c in all_curves]))
    return curves

def __get_all_wells():
    base_dir = Naming.well_path()
    return [os.path.basename(f) for f in iglob(f'{base_dir}/*')]

def get_wells_has_curve(curve: str):
    storage = Store()
    all_wells = __get_all_wells()
    matched_wells = []
    for w in all_wells:
        curves = get_curves_in_well(w)
        sim_curves = find_similar_curves(curve, curves)
        if len(sim_curves):
            matched_wells.append({'well': w, 'curves': sim_curves, 'all_curves': curves})

    return matched_wells

def get_wells_has_markers():
    storage = Store()
    all_wells = __get_all_wells()
    matched_wells = []
    for w in all_wells:
        _, zones = XLSX.extract_zones1(w)
        if zones is not None and len(zones.index):
            curves = get_curves_in_well(w)
            matched_wells.append({'well': w, 'curves': ['Zone'], 'all_curves': curves + ['Zone']})

    return matched_wells

def _read_missing_pay_data(well):
    cached_file = f'missing_pay_cache/{well}.csv'
    storage = Store()

    df_result = storage.load(cached_file)
    if df_result is not None:
        return df_result    

    df = read_curves_from_las(well, ['TVDSS', 'PAYF'])

    index_col = df.index.name or 'index'
    df = df.reset_index()
    target_series = df[index_col]

    try:
        # perforation
        perforationDF = XLSX.extract_perforation_curve(well, target_series)
        df['PERF'] = perforationDF['PERF']
    except Exception as e:
        print(e)
        df['PERF'] = None
    df = df.set_index(index_col)
    # zones
    keyzone, zone=XLSX.extract_zones1(well)
    if zone is not None:
        zone['DEPTH'] = zone['start'].apply(lambda x: target_series.iloc[(abs(target_series - x)).idxmin()])
        zone = zone.set_index('DEPTH')
        zone = zone[~zone.index.duplicated(keep='first')]
        df['Zone'] = zone['Surface']
        df['Zone'] = df['Zone'].ffill()
    else:
        df['Zone'] = None

    sim_curves = find_similar_curves('PAYF', list(df.columns))
    if sim_curves is None or len(sim_curves) == 0: # No PAYF flag exist
        df['PAYF'] = np.nan
        sim_curves = ['PAYF']
    state = dict(cur_zone=None, prev_sample=0, pay_cnt=0)
    df['PAY_NAME'] = df.apply(lambda row: mark_sample(row, state, payfname = sim_curves[-1]), axis=1)
    df = df.reset_index()
    df = df.rename(columns={index_col: 'MD'})

    df_perf = df.groupby('PAY_NAME')[['PERF']].mean()

    df_min = df.groupby('PAY_NAME')[['MD', 'TVDSS']].min()
    df_max = df.groupby('PAY_NAME')[['MD', 'TVDSS']].max()
    df_net = df_max - df_min

    df_result = df_min[["MD"]]
    df_result.columns = ['Top_MD']
    df_result['Bottom_MD'] = df_max['MD']
    df_result['GrossNET_MD'] = df_net['MD']

    df_result['Top_TVDSS'] = df_min['TVDSS']
    df_result['Bottom_TVDSS'] = df_max['TVDSS']
    df_result['GrossNET_TVDSS'] = df_net['TVDSS']

    df_result['PERF'] = df_perf['PERF']

    df_result = df_result.reset_index()
    
    storage.save(df_result, cached_file)

    return df_result[df_result['GrossNET_MD'] > 0]

def read_missing_pay_data(wells):
    if wells is None or len(wells) == 0:
        wells = __get_all_wells()
    dfs = dict()
    for well in wells:
        df = _read_missing_pay_data(well)
        dfs[well] = df
        print('_read_missing_pay_data for well', well, list(df.columns))

    return dfs

############ Clastic Interpretation ################

gr_clean, gr_clay = 40, 135
sp_clean, sp_clay = -60,2

neut_clean1, den_clean1 = 15, 2.6
neut_clean2, den_clean2 = 40, 2
neut_clay, den_clay =47.5, 2.8

#VCLGR
def vclgr(gr_log, gr_clean, gr_clay, correction=None):

    igr=(gr_log - gr_clean)/(gr_clay - gr_clean)       #Linear Gamma Ray

    if correction == "young":
        vclgr_larionov_young= 0.083*( 2**(3.7*igr) - 1 )   #Larionov (1969) - Tertiary rocks
        vclgr=vclgr_larionov_young
    elif correction == "older":
        vclgr_larionov_old = 0.33*( 2**(2*igr) - 1 )        #Larionov (1969) - Older rocks
        vclgr = vclgr_larionov_old
    elif correction=="clavier":
        vclgr_clavier = 1.7 - (3.38 - (igr + 0.7)**2)**0.5    #Clavier (1971)
        vclgr=vclgr_clavier
    elif correction=="steiber":
        vclgr_steiber = 0.5*igr/(1.5 - igr)               #Steiber (1969) - Tertiary rocks
        vclgr=vclgr_steiber
    else:
        vclgr=igr
    return vclgr

#VCLSP
def vclsp(sp_log, sp_clean, sp_clay):
    vclsp=(sp_log - sp_clean)/(sp_clay - sp_clean)
    return vclsp

#VCLRT
def vclrt(rt_log, rt_clean,rt_clay):
    vrt=(rt_clay/rt_log)*(rt_clean - rt_log)/(rt_clean - rt_clay)
    if (rt_log > 2* rt_clay):
        vclrt = 0.5 * (2 * vrt)** (0.67*(vrt + 1)) 
    else:
        vclrt = vrt
    return vclrt

#VCLND
def vclnd(neut_log,den_log,neut_clean1,den_clean1,neut_clean2,den_clean2,neut_clay,den_clay):
    term1 = (den_clean2 - den_clean1)*(neut_log - neut_clean1)-(den_log - den_clean1)*(neut_clean2 - neut_clean1)
    term2 =(den_clean2 - den_clean1)*(neut_clay - neut_clean1)-(den_clay - den_clean1)*(neut_clean2 - neut_clean1)
    vclnd=term1/term2
    return vclnd


