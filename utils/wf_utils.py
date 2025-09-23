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
#from pywaterflood import CRM
from utils.crm import CRM
from base_utils import compute_metrics, get_mlflow_experiment, human_readable_diff, parse_json_param, parse_float_param, get_mlflow_artifact_path, seconds_ago_to_timestamp

from utils.plot_utils import multi_chart
from naming import Naming

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def getdaysofmonth(datestr:str):
     dt = datetime.strptime(datestr, '%Y-%m-%d')
     _, daynum = monthrange(dt.year, dt.month)
     return daynum

def build_wf_input(iwells:list[str], owells:list[str], production_col=2, injection_col=5) -> pd.DataFrame:
    #df = XLSX.extract_production_data([*iwells, *owells], idxcols=[0, 1, 6, 10, 12, 14], colnames = None)
    df = XLSX.extract_production_data([*iwells, *owells], idxcols=[0, 1, 7, 10, 12, 15], colnames = None)
    print('------', df.columns)
    #df = df.dropna()
    df = df.fillna(0)
    print("***********", df.Date.unique())
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
        df[colName] = df[colName] * 1000
        if merged_df is None:
            merged_df = df[['Date', colName]]
        else:
            print(">>>", merged_df.columns)
            print("<<<", df.columns)
            merged_df = pd.merge(merged_df, df[['Date', colName]], on='Date', how='outer')

    for w in iwells:
        colName = f"I{w.strip()}"
        colMapping = { dfs[w].columns[injection_col]: colName }
        df = dfs[w].rename(columns=colMapping)
        df[colName] = df[colName] * 1000
        if merged_df is None:
            merged_df = df[['Date', colName]]
        else:
            merged_df = pd.merge(merged_df, df[['Date', colName]], on='Date', how='outer')

    merged_df['Time'] = merged_df['Date'].apply(getdaysofmonth).cumsum()
    merged_df.to_csv('/tmp/crm_input.csv')
    return merged_df

def build_wf_input_for_reservoir(reservoir: str):
    df = XLSX.extract_production_data(idxcols=[0, 1, 6, 10, 12, 14, 22, 23, 25], colnames = None)
    if reservoir not in df['Completion'].unique():
        raise Exception(f"Reservoir {reservoir} not found")

    df = df[ df['Completion'] == reservoir ]

    owells = list(df[df["CV.WellProd"] == 1]['Master.Wellnumber'].unique())
    iwells = list(df[df["CV.WellInj"] == 1]['Master.Wellnumber'].unique())
    print("owells", owells)
    print("iwells", iwells)
    return build_wf_input(iwells, owells)

def __do_crm_train(df, production_wells, injection_wells, tau_selection=None, constraints=None, mode='', cutoff=False, train_ratio = 0.8):
    models = {}
    for idx, owell in enumerate(production_wells):
        if mode == 'P':
            crm = CRM(tau_selection=tau_selection, constraints=constraints, N_inj=len(injection_wells), I_P=False)
        elif mode == 'IP':
            crm = CRM(tau_selection=tau_selection, constraints=constraints, N_inj=len(injection_wells), I_P=True)
        else:
            crm = CRM(tau_selection=tau_selection, constraints=constraints)
        start_idx = 0
        if cutoff:
            prod_idx = (df[owell] > 0).idxmax()
            inj_idx = (df[injection_wells].sum(axis=1) > 0).idxmax()
            start_idx = max(prod_idx, inj_idx)
        df_cut = df.iloc[start_idx:, :]
        dfsize = len(df_cut.index)
        df_train_size = round(dfsize * train_ratio)
        test_idx = start_idx + df_train_size

        prod_arr = df[[ owell ]].values[start_idx:test_idx, :]
        np.savetxt(f'/tmp/{owell}.txt', prod_arr, delimiter=',')
        inj_arr = df[injection_wells].values[start_idx:test_idx,:]
        np.savetxt(f'/tmp/i{owell}.txt', inj_arr, delimiter=',')
        time_arr = df['Time'].astype(np.float64).values[start_idx:test_idx]
        np.savetxt(f'/tmp/t{owell}.txt', time_arr, delimiter=',')
        crm.fit(df[[ owell ]].values[start_idx:test_idx, :], 
                df[injection_wells].values[start_idx:test_idx,:], 
                df['Time'].astype(np.float64).values[start_idx:test_idx])
        models[owell] = {'model': crm, 'start_idx': start_idx, 'test_idx': test_idx}
    return models

def __do_crm_self_predict(df, crm_models, injection_wells, production_wells):
    train_qs = []
    test_qs = []
    for idx,owell in enumerate(production_wells):
        crm = crm_models[owell]['model']
        start_idx = crm_models[owell]['start_idx']
        test_idx = crm_models[owell]['test_idx']
        train_q = crm.predict(injection=df[injection_wells].values[start_idx:test_idx, :], time=df['Time'].astype(np.float64).values[start_idx:test_idx])
        test_q = crm.predict(injection=df[injection_wells].values[test_idx:, :], time=df['Time'].astype(np.float64).values[test_idx:])
        train_qs.append(train_q)
        test_qs.append(test_q)
    return np.hstack(train_qs),np.hstack(test_qs)

def __do_crm_predict(crm_models, injection, time, production_wells):
    qs = []
    for idx,owell in enumerate(production_wells):
        crm = crm_models[owell]['model']
        q_pred = crm.predict(injection=injection, time=time)
        qs.append(q_pred)
    return np.hstack(qs)

def train_crm(df, experiment, run, tau_selection = 'per-pair', constraints = 'up-to one', mode='', cutoff=False):
    mlflow.log_param("tau_selection", tau_selection)
    mlflow.log_param("constraints", constraints)

    columns = list(df.columns)
    production_wells = [ w for w in columns if w.startswith("P") ]
    injection_wells = [ w for w in columns if w.startswith("I") ]

    crm_models = __do_crm_train(df, production_wells, injection_wells, tau_selection=tau_selection, constraints=constraints, mode=mode, cutoff=cutoff)

    q_train, q_test = __do_crm_self_predict(df, crm_models, injection_wells, production_wells)
    # Future injection
    start_date = df['Date'].values[-1]
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime('2033-12-01', '%Y-%m-%d')
    
    lastTime = df['Time'].values[-1]
    df_specs = {"Date": pd.date_range(start=start_date, end=end_date, freq='MS')}
    for iwell in injection_wells:
        df_specs[iwell] = df[iwell].values[-1]
    df_future = pd.DataFrame(df_specs)
    df_future = df_future.iloc[1:].reset_index(drop=True)
    df_future['Date'] = df_future['Date'].astype(str)
    df_future['Time'] = df_future['Date'].apply(getdaysofmonth).cumsum() + lastTime
    print(df_future)
    q_future = __do_crm_predict(crm_models, df_future[injection_wells].values, df_future['Time'].astype(np.float64).values, production_wells)

    metrics_df = pd.DataFrame()
    well_links = []
    MAEs = []
    RMSEs = []
    MAPEs = []
    R2s = []

    data4 = dict(x=df['Date'].values, y=df[injection_wells].values)
    q_real = df[production_wells].values
    for idx,owell in enumerate(production_wells):
        start_idx = crm_models[owell]['start_idx']
        test_idx = crm_models[owell]['test_idx']
        data1 = dict(x=df['Date'].values[start_idx:], y=q_real[start_idx:, idx].reshape(-1,1))
        data2 = dict(x=df["Date"].values[start_idx:test_idx], y=q_train[:, idx].reshape(-1, 1))
        data3 = dict(x=df["Date"].values[test_idx:], y=q_test[:, idx].reshape(-1,1))
        dataF = dict(x=df_future['Date'].values, y = q_future[:, idx].reshape(-1,1))
        fig = multi_chart([owell], [data1, data2, data3, dataF], injection_wells, [data4], main_title = f"Summarization CRM mode={mode} cutoff={cutoff}")
        dest_path = Naming.dest_path(f'crm-chart_{owell}', category='wf-crm')
        plot = fig.write_html(dest_path, include_plotlyjs="/js/plotly-3.0.1.min.js")
        
        mlflow.log_artifact(dest_path, artifact_path='report')
        well_links.append(f"<a href='crm-chart_{owell}.html' target='_blank'>{owell}</a>")
        
        metrics = compute_metrics(df[owell].values[start_idx:test_idx], q_train[:, idx])
        MAEs.append(metrics['MAE'])
        RMSEs.append(metrics['RMSE'])
        MAPEs.append(metrics['MAPE'])
        R2s.append(metrics['R2'])
    
    metrics_df = pd.DataFrame({"Well": well_links, "MAE": MAEs, "RMSE": RMSEs, "MAPE": MAPEs, 'R²': R2s})
    table = metrics_df.to_html(index=False, escape=False)
    #data1 = dict(x=df['Date'].values, y=df[production_wells].values)
    #data2 = dict(x=df_train["Date"].values, y=q_train)
    #data3 = dict(x=df_validate["Date"].values, y=q_test)
    #dataF = dict(x=df_future['Date'].values, y = q_future)
    #data4 = dict(x=df['Date'].values, y=df[injection_wells].values)
    #fig = multi_chart(production_wells, [data1, data2, data3, dataF], injection_wells, [data4])

    dest_path = Naming.dest_path('crm-chart', category='wf-crm')
    publish_path = Naming.publish_path('crm-chart', category='wf-crm')

    template_path = "templates/wf_training_result_report_tpl.html"
    template = None
    with open(template_path, "r") as tpl_file:
        template = tpl_file.read()
    
    report_html = template.replace("{{TABLE}}", table).replace("{{PLOT}}", "")
    with open(dest_path, "w") as f:
        f.write(report_html)

    mlflow.log_param('report_file', os.path.basename(dest_path))
    mlflow.log_artifact(dest_path, artifact_path='report')
    return get_mlflow_artifact_path(experiment.experiment_id, run.info.run_id, f'report/{os.path.basename(dest_path)}')
    #return publish_path


#Create subdataset (1 Production Wells + 2 Injection Wells)
def create_subdataset_production(df, prod_cols, inj_cols):
    subdataset = {}
    for prod in prod_cols:
        columns_include = ['Date'] + [prod] + inj_cols
        subdataset[prod] = df[columns_include].copy()
        subdataset[prod] = subdataset[prod].dropna(subset=[prod])
        subdataset[prod] = subdataset[prod].reset_index()

        subdataset[prod]['Date'] = pd.to_datetime(subdataset[prod]['Date'])
        start_date = subdataset[prod]['Date'].iloc[0]
        subdataset[prod]['Time'] = (subdataset[prod]['Date'] - start_date).dt.days
        if len(subdataset[prod]) > 0:
            days_in_first_month = (start_date.replace(day=1) + pd.DateOffset(months=1) - start_date).days
            subdataset[prod]['Time'] = subdataset[prod]['Time'] + days_in_first_month

    return subdataset

def create_sequences1(df, cols, seq_len):
    X = df[cols].values
    Xs = []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
    return np.array(Xs)

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def process_subdatasets_to_sequences(processed_subdatasets, productionwells, injectionwells, seq_len):
    sequenced_data = {}
    feat_cols = injectionwells + ['Time'] # Features are injection wells and Time
    for product, subdataset in processed_subdatasets.items():
        if product in productionwells: # Ensure we are processing a production well subdataset
            ## Separate features (X) and target (y)
            #X = subdataset[feat_cols].values
            #y = subdataset[product].values
            
            ## Create sequences
            #Xs, ys = create_sequences(X, y, seq_len)

            sequenced_data[product] = {
                'X': create_sequences1(subdataset, feat_cols, seq_len), 
                'y': create_sequences1(subdataset, product, seq_len)
            }
    return sequenced_data

def train_model_for_well(well_names, sequenced_data, lstm_units, batch_size, epochs, learning_rate):
    trained_models_and_history = {}

    for well_name in well_names:
        if well_name not in sequenced_data:
            print(f"Warning: No sequenced data found for well '{well_name}'. Skipping.")
            continue

        X_seq = sequenced_data[well_name]['X']
        y_seq = sequenced_data[well_name]['y']

        # split train - test set (using the split from the previous code)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )

        model = Sequential([
            LSTM(lstm_units, input_shape=(X_tr.shape[1], X_tr.shape[2])),
            Dense(1)
        ])

        # set up loss funtion - optimizer
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='mae')
        print(f"\nTraining model for well: {well_name}")
        model.summary()

        # training
        history = model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=2
        )

        mlflow.tensorflow.log_model(model, name=f"{well_name}")

        trained_models_and_history[well_name] = {
            'model': model,
            'history': history,
            'X_test': X_te,
            'y_test': y_te,
            'X_train': X_tr,
            'y_train': y_tr
        }
        print(f"Training finished for {well_name}.")

    return trained_models_and_history

def make_predictions(trained_models_info):
    predictions = pd.DataFrame()
    predictions = {}

    for well_name, info in trained_models_info.items():
        model = info['model']
        X_train = info['X_train']
        X_test = info['X_test']

        # Make predictions on the training set
        y_tr_pred = model.predict(X_train)
        print(f"Predictions made for training set of well: {well_name}")

        # Make predictions on the test set
        y_te_pred = model.predict(X_test)
        print(f"Predictions made for test set of well: {well_name}")


        predictions[well_name] = {
            'train_predictions': y_tr_pred,
            'test_predictions': y_te_pred
        }

    return predictions

def train_lstm(df, experiment, run):
    #parameter
    lstm_units    = 128
    batch_size    = 32
    epochs        = 2000
    learning_rate = 7e-4

    seq_len = 1

    columns = list(df.columns)
    production_wells = [ w for w in columns if w.startswith("P") ]
    injection_wells = [ w for w in columns if w.startswith("I") ]

    production_subdatasets = create_subdataset_production(df, production_wells, injection_wells)

    production_sequences = process_subdatasets_to_sequences(production_subdatasets, production_wells, injection_wells, seq_len)
    
    train_models_info = train_model_for_well( production_wells, production_sequences, lstm_units, batch_size, epochs, learning_rate)
    prediction_data = make_predictions(train_models_info)

    # TODO
    start_date = df['Date'].values[-1]
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime('2033-12-01', '%Y-%m-%d')
    
    lastTime = df['Time'].values[-1]
    df_specs = {"Date": pd.date_range(start=start_date, end=end_date, freq='MS')}
    for iwell in injection_wells:
        df_specs[iwell] = df[iwell].values[-1]

    df_future = pd.DataFrame(df_specs)
    df_future = df_future.iloc[1:].reset_index(drop=True)
    df_future['Date'] = df_future['Date'].astype(str)
    df_future['Time'] = df_future['Date'].apply(getdaysofmonth).cumsum() + lastTime
    print(df_future)


    metrics_df = pd.DataFrame()
    well_links = []
    MAEs = []
    RMSEs = []
    MAPEs = []
    R2s = []

    data4 = dict(x=df['Date'].values, y=df[injection_wells].values)
    q_real = df[production_wells].values
    for idx,owell in enumerate(production_wells):
        info = prediction_data[owell]
        y_tr_pred = info['train_predictions']
        y_te_pred = info['test_predictions']

        num_sequenced_samples = len(y_tr_pred) + len(y_te_pred)
        num_train_sequenced_samples = len(y_tr_pred)
        
        all_dates = production_subdatasets[owell]['Date']
        train_dates = all_dates.iloc[seq_len : seq_len + num_train_sequenced_samples].tolist()
        test_dates = all_dates.iloc[seq_len + num_train_sequenced_samples : seq_len + num_sequenced_samples].tolist()

        data1 = dict(x=df['Date'].values, y=q_real[:, idx].reshape(-1,1))
        data2 = dict(x=train_dates, y=y_tr_pred)
        #data2 = dict(x=df['Date'].values, y=prediction_data[owell]['train_predictions'])
        data3 = dict(x=test_dates, y=y_te_pred)
        #data3 = dict(x=df_validate["Date"].values, y=q_test[:, idx].reshape(-1,1))
        
        X_future = create_sequences1(df_future, injection_wells + ['Time'], seq_len)        
        q_future = train_models_info[owell]['model'].predict(X_future)

        dataF = dict(x=df_future['Date'].values, y = q_future)
        #dataF = dict(x=df_future['Date'].values, y = q_future[:, idx].reshape(-1,1))
        fig = multi_chart([owell], [data1, data2, data3, dataF], injection_wells, [data4])
        dest_path = Naming.dest_path(f'lstm-chart_{owell}', category='wf-lstm')
        plot = fig.write_html(dest_path, include_plotlyjs="/js/plotly-3.0.1.min.js")
        
        mlflow.log_artifact(dest_path, artifact_path='report')
        well_links.append(f"<a href='lstm-chart_{owell}.html' target='_blank'>{owell}</a>")
        
        metrics = compute_metrics(train_models_info[owell]['y_train'].reshape(-1), y_tr_pred.reshape(-1))
        MAEs.append(metrics['MAE'])
        RMSEs.append(metrics['RMSE'])
        MAPEs.append(metrics['MAPE'])
        R2s.append(metrics['R2'])
    
    metrics_df = pd.DataFrame({"Well": well_links, "MAE": MAEs, "RMSE": RMSEs, "MAPE": MAPEs, 'R²': R2s})
    table = metrics_df.to_html(index=False, escape=False)
    #data1 = dict(x=df['Date'].values, y=df[production_wells].values)
    #data2 = dict(x=df_train["Date"].values, y=q_train)
    #data3 = dict(x=df_validate["Date"].values, y=q_test)
    #dataF = dict(x=df_future['Date'].values, y = q_future)
    #data4 = dict(x=df['Date'].values, y=df[injection_wells].values)
    #fig = multi_chart(production_wells, [data1, data2, data3, dataF], injection_wells, [data4])

    dest_path = Naming.dest_path('lstm-chart', category='wf-lstm')
    publish_path = Naming.publish_path('lstm-chart', category='wf-lstm')

    template_path = "templates/wf_training_result_report_tpl.html"
    template = None
    with open(template_path, "r") as tpl_file:
        template = tpl_file.read()
    
    report_html = template.replace("{{TABLE}}", table).replace("{{PLOT}}", "")
    with open(dest_path, "w") as f:
        f.write(report_html)

    mlflow.log_param('report_file', os.path.basename(dest_path))
    mlflow.log_artifact(dest_path, artifact_path='report')
    return get_mlflow_artifact_path(experiment.experiment_id, run.info.run_id, f'report/{os.path.basename(dest_path)}')

def wf_filter_params(iwells, owells, model_type, seconds, filter_expr):
    filter_params = []
    
    if iwells and len(iwells):
        filter_params.append(f"params.injection_wells = '{json.dumps(iwells)}'")
    
    if owells and len(owells):
        filter_params.append(f"params.production_wells = '{json.dumps(owells)}'")
   
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
    modes: list[str] = []
    cutoffs: list[str] = []

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
        
        iwells = parse_json_param(data.params.get("injection_wells", "[]"))
        iwells = iwells[:5]
        owells = parse_json_param(data.params.get("production_wells", "[]"))
        owells = owells[:5]
        model = data.params.get("model_type", "CRM")
        mode = data.params.get('mode', "default")
        cutoff = data.params.get('cutoff', False)
        
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
        modes += [mode]
        cutoffs += [cutoff]
        durations += [time_since_run]
        status_list += [run_status]
        
    # generate table
    table_df = pd.DataFrame(dict(ID=run_ids, Model=model_names, Injection=iwells_list, 
                            Production=owells_list, Mode=modes, Cutoff=cutoffs, 
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
