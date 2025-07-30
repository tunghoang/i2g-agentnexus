import os
import numpy as np
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from config.settings import DataConfig
from store import Store
from naming import Naming
from cache import MemoryCache
import pandas as pd
import mlflow
from calendar import monthrange
from pywaterflood import CRM
from utils.plot_utils import multi_chart
from xlsx_utils import XLSX

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
                return {"text": json.dumps(markerDF.to_dict('records'))}

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
            return {"text": json.dumps(filteredDF.to_dict('records'))}
        except Exception as e:
            traceback.print_exc()
            return {"text": "Tool failed: {str(e)}"}

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
    @mcp_server.tool(name="discover_wells_in_prodmonthly", description="Discover wells in production monthly file")
    def discover_wells_in_prodmonthly(input):
        try:
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
            columns = list(df.columns)
            df[columns[well_column]] = df[columns[well_column]].astype(str)
            df = df[[ columns[i] for i in data_columns ]]
            idx = df.groupby(columns[well_column])[columns[data_columns[0]]].idxmax()
            resultDF = df.iloc[idx]
            print(resultDF)
            def conclude_well(row):
                print(row)
                return 'injection' if row[columns[15]] > 0 else 'production'
            resultDF['type'] = resultDF.apply(conclude_well, axis=1)
            resultDF['Date'] = pd.to_datetime(resultDF['Date']).astype(str)
            print(resultDF)
            return {"text": json.dumps(resultDF.to_dict("records"))}
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

    @mcp_server.tool(
        name="trainCRMModel",
        description="Train CRM model using a input file"
    )
    def trainCRMModel(**kwargs):
        CHART_DIR = '/tmp'
        try:
            input_data = json.loads(kwargs['input'])
            filepath = input_data.get('filepath', None)
            if not filepath:
                return {"text": "filepath should not be None or empty" }

            input_file = os.path.join('data/', filepath)

            df = pd.read_csv(input_file, header=0)
            df = df.fillna(0)
            columns = list(df.columns)
            production_wells = [ w for w in columns if w.startswith("P") ]
            injection_wells = [ w for w in columns if w.startswith("I") ]

            train_ratio = 0.8
            dfsize = len(df.index)
            df_train_size = round(dfsize * train_ratio)
            df_train = df.iloc[:df_train_size]
            df_validate = df.iloc[df_train_size:]

            crm = CRM(tau_selection='per-pair', constraints='up-to one')
            crm.fit(df_train[production_wells].values, df_train[injection_wells].values, df_train["Time"].astype(np.float64).values)
            q_train = crm.predict()
            q_test = crm.predict(injection=df_validate[injection_wells].values, time=df_validate['Time'].astype(np.float64).values)

            data1 = dict(x=df['Time'].astype(np.float64).values, y=df[production_wells].values)
            data2 = dict(x=df_train["Time"].astype(np.float64).values, y=q_train)
            data3 = dict(x=df_validate["Time"].astype(np.float64).values, y=q_test)
            fig = multi_chart(production_wells, data1, data2, data3)
            dest_path = os.path.join(CHART_DIR, 'crm-chart.html')
            fig.write_html(dest_path)
            return {'text': 'The result has been generated in crm-chart.html'}
        except Exception as e:
            traceback.print_exc()
            return {"text": "trainCRMModel failed: {str(e)}"}

    @mcp_server.tool(
        name="production_by_time",
        description="Plot production params by time for wells from production data file",
    )
    def production_by_time(input: str) -> dict:
        try:
            input_data = json.loads(input)
            wells: list[str] = input_data["wells"]
            params: list[str] = input_data.get("params")
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
            params_indices = [PARAMS[p] + 6 for p in params]
            DATE_COL = 0
            WELL_COL = 1
            df = XLSX.parse_well_production()
            df = df[[df.columns[c] for c in [DATE_COL, WELL_COL, *params_indices]]]
            cols = df.columns
            if len(wells) > 0:
                df = df[df[df.columns[WELL_COL]].isin(wells)]
            df_wells = [*df.groupby(cols[WELL_COL])]
            if not len(df_wells):
                raise Exception(f"Failed to find {wells} in production data file")
            if not len(wells):
                wells = [str(w) for w, _ in df_wells]

            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=len(df_wells),
                cols=1,
                subplot_titles=[f"{w} Production" for w, _ in df_wells],
                vertical_spacing=0.02,
            )
            import plotly.colors as pc
            colors = pc.qualitative.Plotly[:len(cols[2:])]

            import plotly.graph_objects as go
            for i, (well, df_well) in enumerate(df_wells):
                for j, col in enumerate(cols[2:]):
                    fig.add_trace(
                        go.Scatter(
                            x=df_well[cols[DATE_COL]],
                            y=df_well[col],
                            name=f"{col}",
                            yaxis=f"y{j+1}",
                            mode="lines",
                            showlegend=i == 0,
                            line=dict(color=colors[j]),
                        ),
                        row=i + 1,
                        col=1,
                    )
            fig.update_layout(height=500 * len(df_wells))

            out_file = Naming.sanitize_filename(f"production-time-chart{'-'.join(wells)}{'-'.join(params)}")
            dest_path = Naming.dest_path(out_file)
            fig.write_html(dest_path)
            return {'text': f'The result has been generated in {out_file}'}
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))

    tool_names = [
        "marker4well",
        "productiondata4well",
        "buildCRMInput",
        "trainCRMModel",
        "production_by_time",
    ]

    return tool_names
