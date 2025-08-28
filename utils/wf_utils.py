import os
import pandas as pd
import uuid
import json

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from xlsx_utils import XLSX
from datetime import datetime
from calendar import monthrange
import numpy as np
from pywaterflood import CRM
from base_utils import compute_metrics, get_mlflow_experiment, human_readable_diff, parse_json_param, parse_float_param, get_mlflow_artifact_path, seconds_ago_to_timestamp

from utils.plot_utils import multi_chart
from naming import Naming
def getdaysofmonth(datestr:str):
     dt = datetime.strptime(datestr, '%Y-%m-%d')
     _, daynum = monthrange(dt.year, dt.month)
     return daynum

def build_wf_input(iwells:list[str], owells:list[str], production_col=2, injection_col=5) -> pd.DataFrame:
    df = XLSX.extract_production_data([*iwells, *owells], idxcols=[0, 1, 7, 10, 12, 15], colnames = None)
    print('------', df.columns)
    df.sort_values(by = ['Date'], ascending = [True])
    grouped = df.groupby('Master.Wellnumber')

    dfs = {}
    for name, group_df in grouped:
        dfs[name] = group_df

    merged_df = None
    for w in owells:
        colName = f"P{w.strip()}"
        colMapping = { dfs[w].columns[production_col]: colName }
        df = dfs[w].rename(columns=colMapping)
        if merged_df is None:
            merged_df = df[['Date', colName]]
        else:
            print(merged_df.columns)
            print(df.columns)
            merged_df = pd.merge(merged_df, df[['Date', colName]], on='Date', how='outer')

    for w in iwells:
        colName = f"I{w.strip()}"
        colMapping = { dfs[w].columns[injection_col]: colName }
        df = dfs[w].rename(columns=colMapping)
        if merged_df is None:
            merged_df = df[['Date', colName]]
        else:
            merged_df = pd.merge(merged_df, df[['Date', colName]], on='Date', how='outer')

    merged_df['Time'] = merged_df['Date'].apply(getdaysofmonth).cumsum()
    print(merged_df)
    merged_df.to_csv('/tmp/crm_input.csv')
    return merged_df

def train_crm(df, tau_selection = 'per-pair', constraints = 'up-to one'):
    client = MlflowClient()
    
    columns = list(df.columns)
    production_wells = [ w for w in columns if w.startswith("P") ]
    injection_wells = [ w for w in columns if w.startswith("I") ]

    train_ratio = 0.8
    dfsize = len(df.index)
    df_train_size = round(dfsize * train_ratio)
    df_train = df.iloc[:df_train_size]
    df_validate = df.iloc[df_train_size:]

    crm = CRM(tau_selection=tau_selection, constraints=constraints)
    crm.fit(df_train[production_wells].values, df_train[injection_wells].values, df_train["Time"].astype(np.float64).values)
    q_train = crm.predict()
    q_test = crm.predict(injection=df_validate[injection_wells].values, time=df_validate['Time'].astype(np.float64).values)

    metrics_df = pd.DataFrame({'Well': production_wells})
    MAEs = []
    RMSEs = []
    MAPEs = []
    R2s = []
    for idx,owell in enumerate(production_wells):
        metrics = compute_metrics(df_train[owell].values, q_train[:, idx])
        MAEs.append(metrics['MAE'])
        RMSEs.append(metrics['RMSE'])
        MAPEs.append(metrics['MAPE'])
        R2s.append(metrics['R2'])
    metrics_df = pd.DataFrame({"Well": production_wells, "MAE": MAEs, "RMSE": RMSEs, "MAPE": MAPEs, 'R²': R2s})
    table = metrics_df.to_html(index=False, escape=False)
    data1 = dict(x=df['Date'].values, y=df[production_wells].values)
    data2 = dict(x=df_train["Date"].values, y=q_train)
    data3 = dict(x=df_validate["Date"].values, y=q_test)
    data4 = dict(x=df['Date'].values, y=df[injection_wells].values)
    fig = multi_chart(production_wells, [data1, data2, data3], injection_wells, [data4])

    dest_path = Naming.dest_path('crm-chart', category='wf-crm')
    publish_path = Naming.publish_path('crm-chart', category='wf-crm')
    #fig.write_html(dest_path)
    plot = fig.to_html(full_html=False, include_plotlyjs="/js/plotly-3.0.1.min.js")

    template_path = "templates/wf_training_result_report_tpl.html"
    template = None
    with open(template_path, "r") as tpl_file:
        template = tpl_file.read()
    
    report_html = template.replace("{{TABLE}}", table).replace("{{PLOT}}", plot)
    with open(dest_path, "w") as f:
        f.write(report_html)

    experiment = get_mlflow_experiment(client, name='wf')
    model_name = f"CRM-{uuid.uuid4().hex[:8]}"
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_name) as run:
        mlflow.log_param("injection_wells", json.dumps(injection_wells))
        mlflow.log_param("production_wells", json.dumps(production_wells))
        mlflow.log_param("tau_selection", tau_selection)
        mlflow.log_param("constraints", constraints)
        mlflow.log_param('report_file', os.path.basename(dest_path))
        mlflow.log_artifact(dest_path, artifact_path='report')
    return publish_path

def wf_filter_params(iwells, owells, model_type, seconds, filter_expr):
    filter_params = []
    
    if iwells and len(iwells):
        filter_params.append(f"params.iwells = '{iwells}'")
    
    if owells and len(owells):
        filter_params.append(f"params.owells = '{owells}'")
    
    #if model_type:
    #    filter_params.append(f"params.model_type = '{model_type}'")
    
    if seconds:
        filter_params.append(f"attributes.start_time >= {seconds_ago_to_timestamp(seconds)}")
    
    return " and ".join(filter_params)
def get_wf_run(iwells: list[str], owells:list[str], model_type:str, seconds: int, filter_expr: str, exp_name='wf'):
    client = MlflowClient()
    experiment = get_mlflow_experiment(client, name=exp_name)

    filter_params = wf_filter_params(iwells, owells, model_type, seconds, filter_expr)
    print(filter_params)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], 
        filter_string=filter_params,
        order_by=["start_time DESC"], 
        max_results=10
    )
    run_ids: list[str] = []
    model_names: list[str] = []
    iwells_list: list[str] = []
    owells_list: list[str] = []
    status_list: list[str] = []
    durations: list[str] = []
    tau_selections: list[str] = []
    constraints_list: list[str] = []

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
        
        iwells = parse_json_param(data.params.get("iwells", "[]"))
        owells = parse_json_param(data.params.get("owells", "[]"))
        model = data.params.get("model_type", "CRM")
        tau_selection = data.params.get('tau_selection')
        constraints = data.params.get('constraints')
        
        report_file_link = 'N/A'
        try:
            report_file = os.path.basename(data.params.get("report_file"))
            report_file_path = get_mlflow_artifact_path(experiment.experiment_id, run_id, artifact_path=f"report/{report_file}")
            report_file_link = f"<a href='{report_file_path[1:]}'>{run_id}</a>" if report_file else "N/A"

        except:
            pass
        run_ids += [report_file_link]
        model_names += [run_name]
        iwells_list += [iwells]
        owells_list += [owells]
        tau_selections += [tau_selection]
        constraints_list += [constraints]
        durations += [time_since_run]
        status_list += [run_status]
        
    # generate table
    table_df = pd.DataFrame(dict(ID=run_ids, Model=model_names, Injection=iwells_list, 
                            Production=owells_list, TauSelection=tau_selections, Constraints=constraints_list, 
                            Duration=durations, Status=status_list))

    table = table_df.to_html(index=False, escape=False)
    
    template_path = "templates/wf_report_list_tpl.html"
    template = None
    with open(template_path, "r") as tpl_file:
        template = tpl_file.read()
    
    dest_path = Naming.dest_path('wf_run_list', category='wf', format='html')
    publish_path = Naming.publish_path('wf_run_list', category='wf', format='html')
    report_html = template.replace("{{TABLE}}", table)
    with open(dest_path, "w") as f:
        f.write(report_html)

    return publish_path
