import os
import re
import numpy as np
import json
import uuid
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from multiprocessing import Process, Event
from pytest import param
from sqlalchemy import over
from config.settings import DataConfig
from store import Store
from naming import Naming
from cache import MemoryCache
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from calendar import monthrange
from pywaterflood import CRM
from tools.plot_tools import getColor
from utils.plot_utils import multi_chart, production_by_time_chart, production_by_oilcum_chart, plot_charts, pie_map
from utils.wf_utils import build_wf_input, build_wf_input_for_reservoir, train_crm, train_lstm, get_wf_run
from xlsx_utils import XLSX
from base_utils import iframe, link, excel_link, normalize, PUBLISH_BASE, get_mlflow_experiment

_cACHE = dict()

def getdaysofmonth(datestr:str):
     dt = datetime.strptime(datestr, '%Y-%m-%d')
     _, daynum = monthrange(dt.year, dt.month)
     return daynum

def excel_column_unique(column=0, file_path='misc/Marker.xlsx', sheet = 0, header=0):
    excel_file = None
    if file_path in _cACHE:
        excel_file = _cACHE[file_path]
    else:
        excel_file = pd.ExcelFile(file_path, engine='openpyxl')

    sheetDF = excel_file.parse(sheet, header=header)
    column = str(sheetDF.columns[column])

    return list(sheetDF[column].unique())


def create_vsp_tools(mcp_server, data_config: DataConfig) -> List[str]:
    PRODUCTION_FILEPATH="production/PVT_WellTest_Perforation_WaterAnalysis.xlsx"
    @mcp_server.tool(
        name="marker4well",
        description="get marker for well from marker file"
    )
    def marker4well(**kwargs):
        try:
            global _cACHE
            input_data = json.loads(kwargs['input'])
            marker_file = input_data.get("marker_file")
            marker_file = 'misc/Marker.xlsx' if not marker_file else marker_file
            marker_file = os.path.join(data_config.data_dir, marker_file)
            well = f"{input_data['well']}"
            store = input_data.get('store', 'default')
            storage = Store()
            if storage.exists(Naming.markername(well)):
                markerDF = storage.load(Naming.markername(well))
                markerDF.sort_values(by='MD', ascending=True, inplace=True)
                temp_path = Naming.dest_path(f"well{well}.marker", category='markers', format='xlsx')
                publish_temp_path = Naming.publish_path(f"well{well}.marker", category='markers', format='xlsx')
                markerDF.to_excel(Naming.dest_path(f"well{well}.marker", category='markers', format='xlsx'))
                #return {"text":f'<iframe width="100%" height="1200px" src=\'{PUBLISH_BASE}/{Naming.publish_path("view_plot")}?file={publish_temp_path}\'></iframe>' }
                return { "text": excel_link(publish_temp_path, label="marker file") }
                #return {"text": json.dumps(markerDF.to_dict('records'))}

            print(f"marker_file = {marker_file}; store={store}");
            if not marker_file:
                return { "text": "marker file is empty or None" }
            elif not os.path.exists(marker_file):
                return { "text": f"{marker_file} does not exist" }
            elif not os.path.isfile(marker_file):
                return { "text": f"{marker_file} is not a regular file" }

            excel_file = None
            if marker_file in _cACHE:
                excel_file = _cACHE[marker_file]
            else:
                excel_file = pd.ExcelFile(marker_file, engine='openpyxl')
            allMarkerDF = excel_file.parse(0, header=0)
            columns = list(allMarkerDF.columns)
            firstColumn = columns[0]
            allMarkerDF[firstColumn] = allMarkerDF[firstColumn].astype(str)
            filteredDF = allMarkerDF[allMarkerDF[firstColumn] == well]
            filteredDF.sort_values(by='MD', ascending=True, inplace=True)
            storage.save(filteredDF, f'well{well}.marker.csv')
            temp_path = Naming.dest_path(f"well{well}.marker", category='markers', format='xlsx')
            publish_temp_path = Naming.publish_path(f"well{well}.marker", category='markers', format='xlsx')
            filteredDF.to_excel(Naming.dest_path(f"well{well}.marker", category='markers', format='xlsx'))
            #return {"text":f'<iframe width="100%" height="1200px" src=\'{PUBLISH_BASE}/{Naming.publish_path("view_plot")}?file={publish_temp_path}\'></iframe>' }
            return { "text": iframe(f'{Naming.publish_path("excel-viewer/",format=None)}?file={publish_temp_path}') }
        except Exception as e:
            traceback.print_exc()
            return {"text": f"Tool failed: {str(e)}"}

    @mcp_server.tool(
        name="build_zone",
        description="build zone for well from marker"
    )
    def build_zone(input):
        try:
            pass
        except Exception as e:
            traceback.print_exc()
            return {"text": str(e)}

    @mcp_server.tool(
        name="productiondata4well",
        description="Retrieve production data for well from a monthly production file"
    )
    def productiondata4well(**kwargs):
        # TODO
        try:
            input_data = json.loads(kwargs['input'])
            production_file = input_data.get('file_path')
            production_file = production_file if production_file else PRODUCTION_FILEPATH
            file_path = os.path.join(data_config.data_dir, production_file)
            sheet = input_data.get('sheet', 4)
            well = input_data.get('well')
            if not well:
                raise Exception("Well should not be None or empty")
            well = well.strip()
            well_column = 1
            data_columns = [
                    0, # Data
                    1, # Master.Wellnumber
                    7, # Monthlyprod.Qoil/1000
                    10,# Monthlyprod.Qwater/1000
                    12,# Monthlyprod.Qgas/1000
                    15,# Monthlyinj.Qwater/1000
            ]

            print(well, file_path, sheet, well_column, input_data)
            storage = Store()
            if storage.exists(Naming.productionRecordName(well)):
                df = storage.load(Naming.productionRecordName(well))
                df.sort_values(by='Date', ascending=True, inplace=True)
                return {"text": json.dumps(df.to_dict('records'))}

            if not well:
                return {"text": "You should specify a well"}
            if not os.path.isfile(file_path):
                return {"text": f"production file {file_path} does not exist or is not a regular file"}
            excel_file = MemoryCache.get_instance().get(production_file)
            if excel_file is None:
                excel_file = pd.ExcelFile(file_path, engine="openpyxl")
                MemoryCache.get_instance().put(production_file, excel_file)

            df = excel_file.parse(sheet, header=0)
            columns = list(df.columns)
            df[columns[well_column]] = df[columns[well_column]].astype(str)
            df = df[df[columns[well_column]] == well]
            df = df[[ columns[i] for i in data_columns ]]
            # df['Date'] = df['Date'].astype(int)
            print(df['Date'])
            print('----------------')
            # df['Date'] = pd.to_datetime(df['Date'], unit='D', origin='1899-12-30').astype(str)
            df['Date'] = pd.to_datetime(df['Date']).astype(str)
            storage.save(df, Naming.productionRecordName(well))
            df.sort_values(by='Date', ascending=True, inplace=True)
            return {"text": json.dumps(df.to_dict("records"))}
        except Exception as e:
            traceback.print_exc()
            return {"text": "Tool failed: {str(e)}"}

    @mcp_server.tool(
        name="describe_production_data",
        description="Describe production data for input wells"
    )
    def describe_production_data(input: str):
        try:
            input_data = json.loads(input)
            wells: list[str] = input_data
            wells = [w.strip() for w in wells]
            wells = [*set(wells)]
            # Metrics
            PARAMS = {  # base Idx = 6
                "CV.OilRate": 0,  # Oil rate
                "Monthlyprod.Qoil/1000": 1,  # Monthly oil rate in thousands
                "CV.Oilcum/1000": 2,  # Oilcum in thousands
                "CV.LiqRate": 3,  # Liquid rate (oil + water)
                "Monthlyprod.Qwater/1000": 4,  # Monthly
                "CV.WaterProdCum/1000": 5,
                "Monthlyprod.Qgas/1000": 6,
                "CV.GasCum/1000": 7,
                "CV.WaterInj_Rate": 8,
                "Monthlyinj.Qwater/1000": 9,
                "CV.WaterInjCum/1000": 10,
                "CV.Watercut": 11,
                "Monthlyprod.Qwater/1000+Monthlyprod.Qoil/1000": 12,
                "Monthlyprod.Qgas/Monthlyprod.Qoil*1000": 13,
                "Monthlyprod.Gor": 14,
                "Monthlyprod.Dayon": 15,
                "CV.WellProd": 16,
                "CV.WellInj": 17,
            }
            DATE_COL = 0
            WELL_COL = 1
            df = XLSX.parse_well_production()
            df[df.columns[WELL_COL]] = df[df.columns[WELL_COL]].astype(str)

            df = df[ [df.columns[WELL_COL], df.columns[DATE_COL]] + list(PARAMS.keys()) ]
            df = df[ df[df.columns[0]].isin(wells) ]
            wellDFs = {well: df[df[df.columns[0]] == well].describe(include='all') for well in wells}

            dest_path = Naming.dest_path('_'.join(wells), category='describe', format='xlsx')
            publish_path = Naming.publish_path('_'.join(wells), category='describe', format='xlsx')
            XLSX.write_excel(wellDFs, dest_path, index=True)
            return {"text": excel_link(publish_path, label="Result")}
        except Exception as e:
            traceback.print_exc()
            return {"text": str(e), "isError": True}
            
    

    @mcp_server.tool(name="discover_wells_in_prodmonthly", description="Discover wells in production monthly file")
    def discover_wells_in_prodmonthly(input):
        try:
            '''
            well_column = 1
            data_columns = [
                    0, # Date
                    1, # Master.Wellnumber
                    7, # Monthlyprod.Qoil/1000
                    10,# Monthlyprod.Qwater/1000
                    12,# Monthlyprod.Qgas/1000
                    15,# Monthlyinj.Qwater/1000
            ]
            # input_data = json.loads(input)
            production_file = PRODUCTION_FILEPATH
            sheet = 4
            excel_file = MemoryCache.get_instance().get(production_file)
            if excel_file is None:
                file_path = Naming.data_path(production_file)
                excel_file = pd.ExcelFile(file_path, engine="openpyxl")
                MemoryCache.get_instance().put(production_file, excel_file)

            df = excel_file.parse(sheet, header=0)
            '''
            well_column = 1
            df = XLSX.extract_production_data()
            columns = list(df.columns)
            df[columns[well_column]] = df[columns[well_column]].astype(str)
            idx = df.groupby(columns[well_column])['Date'].idxmax()
            resultDF = df.iloc[idx]
            print(resultDF)
            def conclude_well(row):
                print(row)
                return 'injection' if row[columns[8]] > 0 else 'production'
            resultDF['type'] = resultDF.apply(conclude_well, axis=1)
            resultDF['Date'] = pd.to_datetime(resultDF['Date']).astype(str)
            print(resultDF)
            dest_path = Naming.dest_path('well_table', format='xlsx', category='production')
            publish_path = Naming.publish_path('well_table', format='xlsx', category='production')
            XLSX.save_dataframe(resultDF, dest_path)
            return {"text": excel_link(publish_path)}
        except Exception as e:
            traceback.print_exc()
            return {"text": str(e), "isError": True}
    @mcp_server.tool(
        name="buildCRMInput",
        description="Build CRM input from a monthly production file and list of wells"
    )
    def buildCRMInput(**kwargs):
        production_col = 2
        injection_col = 5
        input_data = json.loads(kwargs['input'])
        production_wells = input_data.get('production_wells', [])
        injection_wells = input_data.get('injection_wells', [])
        if type(production_wells) == str:
            production_wells = production_wells.split('/,;/')
        if type(injection_wells) == str:
            injection_wells = injection_wells.split('/,;/')

        for w in production_wells:
            productiondata4well(input=json.dumps({"well": w}))

        for w in injection_wells:
            productiondata4well(input=json.dumps({"well": w}))

        storage = Store()
        dfs = []
        for w in production_wells:
            df = storage.load(Naming.productionRecordName(w))
            print(df.columns[production_col], df.columns)
            print(df)
            print("----------------------")
            df1 = df[['Date', df.columns[production_col]]]
            df1 = df1.rename(columns={ df.columns[production_col]: f"P{w.strip()}" })
            dfs.append(df1)
        for w in injection_wells:
            df = storage.load(Naming.productionRecordName(w))
            print(df.columns[injection_col], df.columns)
            print(df)
            print("----------------------")
            df1 = df[['Date', df.columns[injection_col]]]
            df1 = df1.rename(columns={ df.columns[injection_col]: f"I{w.strip()}" })
            dfs.append(df1)

        merged_df = None
        for df in dfs:
            print(df.columns)
            print(df)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on="Date", how="outer")

        merged_df['Time'] = merged_df['Date'].apply(getdaysofmonth).cumsum()

        result_file = "./data/crm_input.csv"
        merged_df.to_csv("./data/crm_input.csv")
        return {"text": result_file}

    def do_train_crm_model(iwells, owells, started_event):
        client = MlflowClient()
    
        experiment = get_mlflow_experiment(client, name='wf')
        model_name = f"CRM-{uuid.uuid4().hex[:8]}"

        run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_name) 

        mlflow.log_param("injection_wells", json.dumps(iwells))
        mlflow.log_param("production_wells", json.dumps(owells))

        if started_event:
            started_event.set()

        df = build_wf_input(iwells, owells)
        df = df.fillna(0)
        train_crm(df, experiment, run, 'per-pair', 'up-to one')

        mlflow.end_run()

    def do_train_crm_model_for_reservoir(reservoir, started_event):
        client = MlflowClient()
    
        experiment = get_mlflow_experiment(client, name='wf')
        model_name = f"CRM-{uuid.uuid4().hex[:8]}"

        run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_name) 

        mlflow.log_param("injection_wells", json.dumps([]))
        mlflow.log_param("production_wells", json.dumps([]))

        if started_event:
            started_event.set()
        df = build_wf_input_for_reservoir(reservoir)
        df = df.fillna(0)
        train_crm(df, experiment, run, 'per-pair', 'up-to one')
        mlflow.end_run()

    @mcp_server.tool(
        name="train_crm_model",
        description="Train CRM model from injection wells and production wells"
    )
    def train_crm_model(input):
        try:
            input_data = json.loads(input)
            iwells = input_data['i_wells']
            owells = input_data['o_wells']

            # start training
            started_event = Event()
            process = Process(
                target=do_train_crm_model,
                args=(
                    iwells, owells,
                    started_event,
                )
            )
            process.start()
            # wait for the training process to init
            started_event.wait()
            
            # view training result
            out_file_relative_path = get_wf_run([], [], 'CRM', 60, "")


            #df = build_wf_input(iwells, owells)

            #df = df.fillna(0)
            #publish_path = train_crm(df, 'per-pair', 'up-to one')
            return {'text': iframe(out_file_relative_path, height="480px")}

        except Exception as e:
            traceback.print_exc()
            return {"text": str(e)}
            
    def do_train_lstm_model(iwells, owells, started_event):
        client = MlflowClient()
    
        experiment = get_mlflow_experiment(client, name='wf')
        model_name = f"LSTM-{uuid.uuid4().hex[:8]}"

        run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_name) 

        mlflow.log_param("injection_wells", json.dumps(iwells))
        mlflow.log_param("production_wells", json.dumps(owells))

        if started_event:
            started_event.set()

        df = build_wf_input(iwells, owells)
        df = df.fillna(0)
        train_lstm(df, experiment, run)

        mlflow.end_run()

    @mcp_server.tool(
        name="train_lstm_model",
        description="Train LSTM model from injection wells and production wells"
    )
    def train_lstm_model(input):
        try:
            input_data = json.loads(input)
            iwells = input_data['i_wells']
            owells = input_data['o_wells']

            # start training
            started_event = Event()
            process = Process(
                target=do_train_lstm_model,
                args=(
                    iwells, owells,
                    started_event,
                )
            )
            process.start()
            # wait for the training process to init
            started_event.wait()
            
            # view training result
            out_file_relative_path = get_wf_run([], [], 'CRM', 60, "")


            #df = build_wf_input(iwells, owells)

            #df = df.fillna(0)
            #publish_path = train_crm(df, 'per-pair', 'up-to one')
            return {'text': iframe(out_file_relative_path, height="480px")}

        except Exception as e:
            traceback.print_exc()
            return {"text": str(e)}
    @mcp_server.tool(
        name="train_crm_model_for_reservoir",
        description="Train CRM model using reservoir name"
    )
    def train_crm_model_for_reservoir(input):
        try:
            input_data = json.loads(input)
            reservoir = input_data.get('reservoir', None)
            # start training
            started_event = Event()
            process = Process(
                target=do_train_crm_model_for_reservoir,
                args=(
                    reservoir,
                    started_event,
                )
            )
            process.start()
            # wait for the training process to init
            started_event.wait()
            
            # view training result
            out_file_relative_path = get_wf_run([], [], 'CRM', 60, "")
            return {'text': iframe(out_file_relative_path, height="480px")}
        except Exception as e:
            traceback.print_exc()
            return {"text": str(e)}

    @mcp_server.tool(
        name="view_wf_experiment",
        description="view_wf_experiment"
    )
    def view_wf_experiment(input: str) -> dict:
        try:
            input_data = json.loads(input)
            iwells: list[str] = input_data.get("iwells")
            owells: list[str] = input_data.get("owells")
            model_type: str = input_data.get("model_type")
            seconds: int = input_data.get("seconds")
            filter_expr: str = input_data.get("filter_expr")

            out_file_relative_path = get_wf_run(iwells, owells, model_type, seconds, filter_expr)

            return {"text": iframe(out_file_relative_path, height='500px')}
        except Exception as e:
            traceback.print_exc()
            return {"text": str(e)}

    @mcp_server.tool(
        name="production_by_time",
        description="Plot production params by time for wells from production data file",
    )
    def production_by_time(input: str) -> dict:
        try:
            input_data = json.loads(input)
            wells: list[str] = [*set(input_data["wells"])]
            params: set[str] = set(input_data.get("params"))
            modes: set[str] = set(input_data.get("modes"))
            # Metrics
            PARAMS = {  # base Idx = 6
                "CV.OilRate": 0,  # Oil rate
                "Monthlyprod.Qoil/1000": 1,  # Monthly oil rate in thousands
                "CV.Oilcum/1000": 2,  # Oilcum in thousands
                "CV.LiqRate": 3,  # Liquid rate (oil + water)
                "Monthlyprod.Qwater/1000": 4,  # Monthly
                "CV.WaterProdCum/1000": 5,
                "Monthlyprod.Qgas/1000": 6,
                "CV.GasCum/1000": 7,
                "CV.WaterInj_Rate": 8,
                "Monthlyinj.Qwater/1000": 9,
                "CV.WaterInjCum/1000": 10,
                "CV.Watercut": 11,
                "Monthlyprod.Qwater/1000+Monthlyprod.Qoil/1000": 12,
                "Monthlyprod.Qgas/Monthlyprod.Qoil*1000": 13,
                "Monthlyprod.Gor": 14,
                "Monthlyprod.Dayon": 15,
                "CV.WellProd": 16,
                "CV.WellInj": 17,
            }
            COLORS = {
                "CV.OilRate": "#ff0000",
                "CV.LiqRate": "#008000",
                "CV.Watercut": "#0000ff",
                "CV.Oilcum/1000": "#800000",
            }
            params_indices = [PARAMS[p] + 6 for p in params]
            DATE_COL = 0
            WELL_COL = 1
            df = XLSX.parse_well_production()
            df = df[[df.columns[c] for c in [DATE_COL, WELL_COL, *params_indices]]]
            all_cols = df.columns
            cols = all_cols[2:]
            if len(wells) > 0:
                df = df[df[df.columns[WELL_COL]].isin(wells)]
            df_wells = [*df.groupby(all_cols[WELL_COL])]
            if not len(df_wells):
                raise Exception(f"Failed to find {wells} in production data file")
            if not len(wells):
                wells = [str(w) for w, _ in df_wells]

            fig = production_by_time_chart(df_wells, all_cols, modes=modes)
            out_file = Naming.sanitize_filename(f"{'-'.join(wells)}{'-'.join(params)}")
            dest_path = Naming.dest_path(out_file, "production-time-chart")
            fig.write_html(dest_path)
            return {'text': Naming.publish_path(out_file, "production-time-chart")}
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))
    @mcp_server.tool(
        name="production_crossplot",
        description="Plot production crossplot for wells from production data file with specified xparam",
    )
    def production_crossplot(input: str) -> dict:
        try:
            input_data = json.loads(input)
            wells: list[str] = [*set(input_data["wells"])]
            params: set[str] = set(input_data.get("params"))
            xparam: str = input_data.get("xparam")
            modes: set[str] = set(input_data.get("modes"))
            if not len(wells):
                wells = [str(w) for w, _ in df_wells]
            # Metrics
            PARAMS = {  # base Idx = 6
                "CV.OilRate": 0,  # Oil rate
                "Monthlyprod.Qoil/1000": 1,  # Monthly oil rate in thousands
                "CV.Oilcum/1000": 2,  # Oilcum in thousands
                "CV.LiqRate": 3,  # Liquid rate (oil + water)
                "Monthlyprod.Qwater/1000": 4,  # Monthly
                "CV.WaterProdCum/1000": 5,
                "Monthlyprod.Qgas/1000": 6,
                "CV.GasCum/1000": 7,
                "CV.WaterInj_Rate": 8,
                "Monthlyinj.Qwater/1000": 9,
                "CV.WaterInjCum/1000": 10,
                "CV.Watercut": 11,
                "Monthlyprod.Qwater/1000+Monthlyprod.Qoil/1000": 12,
                "Monthlyprod.Qgas/Monthlyprod.Qoil*1000": 13,
                "Monthlyprod.Gor": 14,
                "Monthlyprod.Dayon": 15,
                "CV.WellProd": 16,
                "CV.WellInj": 17,
                "CV.WaterRate": 19 #25 - 6
            }
            COLORS = {
                "CV.OilRate": "#ff0000",
                "CV.LiqRate": "#008000",
                "CV.Watercut": "#0000ff",
                "CV.Oilcum/1000": "#800000",
            }
            params_indices = [PARAMS[p] + 6 for p in params]
            WELL_COL = 1
            df = XLSX.parse_well_production()
            if len(wells) > 0:
                df = df[df[df.columns[WELL_COL]].isin(wells)]
            df['CV.WaterRate'] = df['CV.LiqRate'] - df['CV.OilRate']
            
            col_set = {df.columns[c] for c in [*params_indices]}
            col_set = col_set - {xparam}
            col_list = list(col_set)

            df = df[ [xparam, df.columns[WELL_COL], *col_list] ]
            all_cols = df.columns
            cols = all_cols[2:]

            df_wells = [*df.groupby(all_cols[WELL_COL])]
            if not len(df_wells):
                raise Exception(f"Failed to find {wells} in production data file")

            fig = production_by_oilcum_chart(df_wells, all_cols, modes=modes)
            out_file = Naming.sanitize_filename(f"{'-'.join(wells)}{'-'.join(params)}")
            print(out_file)
            dest_path = Naming.dest_path(out_file, "production-crossplot")
            fig.write_html(dest_path)
            print(Naming.publish_path(out_file, "production-crossplot"))
            return {'text': iframe(f'{Naming.publish_path(out_file, "production-crossplot")}')}
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))
    @mcp_server.tool(
        name="production_by_oilcum",
        description="Plot production params by oilcum for wells from production data file",
    )
    def production_by_oilcum(input: str) -> dict:
        try:
            input_data = json.loads(input)
            wells: list[str] = [*set(input_data["wells"])]
            params: set[str] = set(input_data.get("params"))
            modes: set[str] = set(input_data.get("modes"))
            if not len(wells):
                wells = [str(w) for w, _ in df_wells]
            # Metrics
            PARAMS = {  # base Idx = 6
                "CV.OilRate": 0,  # Oil rate
                "Monthlyprod.Qoil/1000": 1,  # Monthly oil rate in thousands
                "CV.Oilcum/1000": 2,  # Oilcum in thousands
                "CV.LiqRate": 3,  # Liquid rate (oil + water)
                "Monthlyprod.Qwater/1000": 4,  # Monthly
                "CV.WaterProdCum/1000": 5,
                "Monthlyprod.Qgas/1000": 6,
                "CV.GasCum/1000": 7,
                "CV.WaterInj_Rate": 8,
                "Monthlyinj.Qwater/1000": 9,
                "CV.WaterInjCum/1000": 10,
                "CV.Watercut": 11,
                "Monthlyprod.Qwater/1000+Monthlyprod.Qoil/1000": 12,
                "Monthlyprod.Qgas/Monthlyprod.Qoil*1000": 13,
                "Monthlyprod.Gor": 14,
                "Monthlyprod.Dayon": 15,
                "CV.WellProd": 16,
                "CV.WellInj": 17,
                "CV.WaterRate": 19 #25 - 6
            }
            COLORS = {
                "CV.OilRate": "#ff0000",
                "CV.LiqRate": "#008000",
                "CV.Watercut": "#0000ff",
                "CV.Oilcum/1000": "#800000",
            }
            params_indices = [PARAMS[p] + 6 for p in params]
            WELL_COL = 1
            OILCUM_COL = 8
            df = XLSX.parse_well_production()
            if len(wells) > 0:
                df = df[df[df.columns[WELL_COL]].isin(wells)]
            df['CV.WaterRate'] = df['CV.LiqRate'] - df['CV.OilRate']
            df = df[[df.columns[c] for c in [OILCUM_COL, WELL_COL, *params_indices]]]
            all_cols = df.columns
            cols = all_cols[2:]

            df_wells = [*df.groupby(all_cols[WELL_COL])]
            if not len(df_wells):
                raise Exception(f"Failed to find {wells} in production data file")

            fig = production_by_oilcum_chart(df_wells, all_cols, modes=modes)
            out_file = Naming.sanitize_filename(f"{'-'.join(wells)}{'-'.join(params)}")
            print(out_file)
            dest_path = Naming.dest_path(out_file, "production-oilcum-chart")
            fig.write_html(dest_path)
            print(Naming.publish_path(out_file, "production-oilcum-chart"))
            return {'text': iframe(Naming.publish_path(out_file, "production-oilcum-chart"))}
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))

    @mcp_server.tool(
        name="summarize_marker_data",
        description="Summarize production layers for wells using marker file and perforation file",
    )
    def summarize_marker_data(input: str) -> dict:
        def strip_prefix(row):
            return re.sub("^[TB]-", "", row)

        try:
            input_data = json.loads(input)
            wells = input_data.get("wells")
            df = XLSX.parse_marker()
            df = df[df[df.columns[1]].str.startswith('T-') | df[df.columns[1]].str.startswith('B-')]
            df['layer'] = df[df.columns[1]].apply(strip_prefix)

            _wells = wells or pd.Series(df[df.columns[0]].unique()).to_list()

            _wells = [w.strip() for w in _wells]
            
            layers = XLSX.extract_layers()
            results = [ ]
            for l in layers:
                _l = str(l).strip().replace("_", "-")
                row = {"Layers": l}
                for w in _wells:
                    row[w] = ""
                    df1 = df[(df[df.columns[0]] == w) & (df['layer'] == l)]
                    if len(df1.index) > 0:
                        row[w] = 'yes'
                results.append(row)
            resultDF = pd.DataFrame(results)
            print(resultDF)
            outpath = Naming.dest_path('misc/Marker.xlsx')
            resultDF.to_html(outpath)
            return {'text': excel_link(Naming.publish_path('misc/Marker.xlsx')) }
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))
    @mcp_server.tool(
        name="production_monthly_data_table",
        description="Show table of production monthly data",
    )
    def production_monthly_data_table(input: str) -> dict:
        try:
            wells = input.split(',')
            wells = [w.strip() for w in wells]
            df = XLSX.extract_production_data(wells)
            _path = Naming.default_production_monthly_file(category='temp')
            XLSX.save_dataframe(df, _path)
            Naming.gen_site()
            return {"text": excel_link(Naming.default_production_monthly_file(category='raw'))}
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))
        
    @mcp_server.tool(
        name="welltest_table",
        description="Show welltest table for wells"
    )
    def welltest_table(input:str) -> dict:
        try:
            input_data = json.loads(input)
            wells = input_data['wells']
            test = input_data['test']
            dfs = XLSX.extract_welltest(wells)
            dest_path = Naming.dest_path("_".join(wells), category='welltest', format="xlsx")
            publish_path = Naming.publish_path("_".join(wells), category='welltest', format="xlsx")
            if test == 'production':
                del dfs['injection']
            elif test == 'injection':
                del dfs['production']
            XLSX.write_excel(dfs, dest_path, index=False)
            return {"text": excel_link(publish_path)}
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))

    @mcp_server.tool(
        name="welltest_chart",
        description="Show welltest chart for wells"
    )
    def welltest_chart(input:str) -> dict:
        try:
            input_data = json.loads(input)
            wells = input_data['wells']
            test = input_data['test']
            dfs = XLSX.extract_welltest(wells)
            dest_path = Naming.dest_path("_".join(wells), category='welltest', format="html")
            publish_path = Naming.publish_path("_".join(wells), category='welltest', format="html")
            if test == 'production':
                del dfs['injection']
            elif test == 'injection':
                del dfs['production']

            fig = plot_charts(dfs)
            fig.write_html(dest_path)
            return {"text": iframe(publish_path, height=f"{480*len(dfs)}px")}
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))
    @mcp_server.tool(
        name="water_analysis_map",
        description="Show water analysis map for input wells"
    )
    def water_analysis_map(input:str) -> dict:
        try:
            input_data = json.loads(input)
            wells = input_data['wells']
            wells = [ w.strip() for w in wells ]
            year = input_data['year']
            month = input_data['month']
            day = input_data['day']

            wellpos_df = XLSX.extract_wellpos(wells)
            df = XLSX.parse_water_analysis()
            #df = XLSX.parse_well_production()
            prod_df = df
            columns = prod_df.columns
            first_col = columns[0]
            if wells and len(wells) > 0:
                prod_df = prod_df[ prod_df['Well'].isin(wells) ]

            #if year:
            #    prod_df = prod_df[ prod_df[prod_df.columns[0]].dt.year == year ]
            date_title = ""
            if year and month and day:
                prod_df = prod_df[ (prod_df[first_col].dt.year <= year) & (prod_df[first_col].dt.month <= month) & (prod_df[first_col].dt.day <= day) ]
                date_title = f"by {year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
            else:
                if year and month:
                    prod_df = prod_df[ (prod_df[first_col].dt.year <= year) & (prod_df[first_col].dt.month <= month) ]
                    date_title = f"in {year}-{str(month).zfill(2)}"
                else: 
                    if year:
                        prod_df = prod_df[ (prod_df[first_col].dt.year <= year) ]
                        date_title = f"in {year}"

            print(date_title)
            prod_df = prod_df[['Well', 'Date', "%WaterProd", '%WaterInj']]
            idx = prod_df.groupby('Well')['Date'].idxmax()
            prod_df = prod_df.loc[idx]

            result_df = pd.merge(prod_df, wellpos_df, on='Well')
            result_df = result_df[['Well', 'X', 'Y', "%WaterProd", '%WaterInj']]
            result_df['%WaterProd'] = result_df['%WaterProd'] / 100
            result_df['%WaterInj'] = result_df['%WaterInj'] / 100

            hovertemplate = [
                '%{customdata[3]:,.1%}   -   %{customdata[4]:,.1%}'
            ]
            hovertemplate = "<br>".join(hovertemplate)
            fig = pie_map(result_df, groups = [[3,4]], names=[['Reservoir water', 'Injection water']], 
                        anno_cols = [3, 4], anno_suffixes = ['', ''],
                        hovertemplate=hovertemplate, 
                        plot_title=f'Water analysis {date_title}')

            dest_path = Naming.dest_path(f'wells_{'_'.join(wells)}', category="water_analysis", format="html")
            publish_path = Naming.publish_path(f'wells_{'_'.join(wells)}', category="water_analysis", format="html")
            fig.write_html(dest_path, config={"showTips": False, "scrollZoom": True})
            return {"text": iframe(f'{publish_path}', height='1200px')}

        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))

    @mcp_server.tool(
        name="production_map",
        description="Show production map for input wells"
    )
    def production_map(input:str) -> dict:
        try:
            input_data = json.loads(input)
            wells = input_data['wells']
            wells = [ w.strip() for w in wells ]
            year = input_data['year']
            month = input_data['month']
            day = input_data['day']
            data = input_data['data']

            wellpos_df = XLSX.extract_wellpos(wells)
            df = XLSX.parse_well_production()
            prod_df = df
            columns = prod_df.columns
            first_col = columns[0]
            if wells and len(wells):
                prod_df = prod_df[ prod_df[prod_df.columns[1]].isin(wells) ]
            #if year:
            #    prod_df = prod_df[ prod_df[prod_df.columns[0]].dt.year == year ]
            date_title = ""
            if year and month and day:
                prod_df = prod_df[ (prod_df[first_col].dt.year <= year) & (prod_df[first_col].dt.month <= month) & (prod_df[first_col].dt.day <= day) ]
                date_title = f"by {year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
            else:
                if year and month:
                    prod_df = prod_df[ (prod_df[first_col].dt.year <= year) & (prod_df[first_col].dt.month <= month) ]
                    date_title = f"in {year}-{str(month).zfill(2)}"
                else: 
                    if year:
                        prod_df = prod_df[ (prod_df[first_col].dt.year <= year) ]
                        date_title = f"in {year}"

            print(date_title)


            selected_columns = ["CV.OilRate", "CV.LiqRate", "CV.Watercut", "CV.WaterInj_Rate", 
                                'CV.Oilcum/1000', 'CV.WaterProdCum/1000', 'CV.WaterInjCum/1000']

            prod_df = prod_df[[columns[0], columns[1], *selected_columns]]
            idx = prod_df.groupby(columns[1])[columns[0]].idxmax()
            idx = idx.dropna()

            prod_df = prod_df.loc[idx]
            prod_df = prod_df.rename(columns={columns[1]: "Well"})

            result_df = pd.merge(prod_df, wellpos_df, on='Well')
            result_df = result_df[['Well', 'X', 'Y', *selected_columns]]
            if data == 'monthly_rate':
                result_df['%OR'] = result_df['CV.OilRate'] / result_df['CV.LiqRate']
                result_df['%WR'] = 1 - result_df['%OR']
                result_df['LiqRateNorm'] = normalize(result_df['CV.LiqRate'])
                result_df['WIRNorm'] = normalize(result_df['CV.WaterInj_Rate'])

                hovertemplate = [
                    'LiqRate & Watercut: %{customdata[4]:,.0f}   -   %{customdata[5]:,.0f}%',
                    'Water Injection Rate: %{customdata[6]:,.0f}'
                ]
                hovertemplate = "<br>".join(hovertemplate)

                fig = pie_map(result_df, groups = [[10,11], []], names=[['Oil rate', 'Water rate'], ['Water inj. rate', '']],
                    anno_cols = [4,3,5,6], anno_suffixes = ['','','%', ''],
                    radius_cols=[12, 13], hovertemplate=hovertemplate, 
                    color_palettes=[
                        ['rgba(154, 205, 50, 0.7)', 'rgba(255, 215, 181, 0.7)'],
                        ['rgba(0, 54, 119, 0.7)', 'blue']
                    ], 
                    plot_title=f'Production map {date_title}'
                )
            elif data == 'cv':
                result_df['%OCV'] = result_df['CV.Oilcum/1000'] / (result_df['CV.Oilcum/1000'] + result_df['CV.WaterProdCum/1000'] )
                result_df['%WCV'] = 1 - result_df['%OCV']
                result_df['LiqCVNorm'] = normalize(result_df['CV.Oilcum/1000'] + result_df['CV.WaterProdCum/1000'])
                result_df['WICVNorm'] = normalize(result_df['CV.WaterInjCum/1000'])

                hovertemplate = [
                    'OilCum/1000 & WaterProdCum/1000: %{customdata[7]:,.0f}   -   %{customdata[8]:,.0f}',
                    'WaterInjCum/1000: %{customdata[9]:,.0f}'
                ]
                hovertemplate = "<br>".join(hovertemplate)

                fig = pie_map(result_df, groups = [[10,11], []], names=[['Oilcum/1000', 'WaterProdCum/1000'], ['WaterInjCum/1000', '']], 
                    anno_cols = [7,8,3,9], anno_suffixes = ['','','%',''],
                    radius_cols=[12, 13], hovertemplate=hovertemplate, 
                    color_palettes=[
                        ['rgba(154, 205, 50, 0.7)', 'rgba(255, 215, 181, 0.7)'],
                        ['rgba(0, 54, 119, 0.7)', 'blue']
                    ], 
                    plot_title=f'Production map {date_title}'
                )
            

            dest_path = Naming.dest_path(f"wells_{'_'.join(wells)}", category="production_map", format="html")
            publish_path = Naming.publish_path(f"wells_{'_'.join(wells)}", category="production_map", format="html")
            fig.write_html(dest_path, config={"showTips": False, "scrollZoom": True})
            return {"text": iframe(f'{publish_path}', height='1200px')}

        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))
    tool_names = [
        "marker4well",
        "productiondata4well",
        "buildCRMInput",
        "train_crm_model",
        "train_crm_model_for_reservoir",
        "train_lstm_model",
        "production_crossplot",
        "production_by_time",
        "production_by_oilcum",
        "summarize_marker_data",
        "production_monthly_data_table",
        "welltest_table",
        "welltest_chart", 
        "water_io_ratio_map",
        "water_analysis_map",
        "production_map"
    ]

    return tool_names
