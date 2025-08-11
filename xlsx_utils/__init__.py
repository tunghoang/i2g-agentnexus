import pandas as pd
import numpy as np
import re
from cache import MemoryCache
from naming import Naming

class XLSX:
    PRODUCTION_FILEPATH="production/PVT_WellTest_Perforation_WaterAnalysis.xlsx"
    PRODUCTION_MONTHLY_SHEET=4

    @classmethod
    def parse_excel(cls, file_path: str, sheet: int = 0):
        excel_file = MemoryCache.get_instance().get(file_path)
        if excel_file is None:
            excel_file = pd.ExcelFile(file_path, engine="openpyxl")
            MemoryCache.get_instance().put(file_path, excel_file)
        df = excel_file.parse(
            sheet,
            header=0,
        )
        return df

    @classmethod
    def parse_well_production(cls, sheet: int = PRODUCTION_MONTHLY_SHEET):
        PROD_WELL_COL = 1
        df = cls.parse_excel(Naming.data_path(cls.PRODUCTION_FILEPATH), sheet)
        #df['Date'] = pd.to_datetime(df['Date'], unit='D', origin='1899-12-30')
        df['Date'] = pd.to_datetime(df['Date'])
        # convert well number to string
        df[df.columns[PROD_WELL_COL]] = df[df.columns[PROD_WELL_COL]].astype(str)
        return df

    @classmethod
    def parse_marker(cls, sheet: int = 0):
        MARKER_WELL_COL = 0
        df = cls.parse_excel(Naming.default_marker_file(), sheet)
        # convert well number to string
        df[df.columns[MARKER_WELL_COL]] = df[df.columns[MARKER_WELL_COL]].astype(str)
        return df

    @classmethod
    def extract_perforation(cls, well):
        PERFORATION_SHEET = 3 # sheet index = 3 (sheet 4)
        WELL_COLUMN = 1
        MD_TOP = 7
        MD_BOTTOM = 8

        xlsx_file = MemoryCache.get_instance().get(cls.PRODUCTION_FILEPATH)
        if xlsx_file is None:
           xlsx_file = pd.ExcelFile(Naming.data_path(cls.PRODUCTION_FILEPATH), engine='openpyxl')
           MemoryCache.get_instance().put(cls.PRODUCTION_FILEPATH, xlsx_file)

        sheetDF = xlsx_file.parse(PERFORATION_SHEET, skiprows=2, header=None)
        dataDF = sheetDF.iloc[:, [1, 7, 8]]
        dataDF.columns.values[0] = 'well'
        dataDF.columns.values[1] = 'start'
        dataDF.columns.values[2] = 'stop'

        dataDF = dataDF[dataDF.well == well]
        if dataDF.empty:
            raise Exception(f'Well {well} does not exist in perforation file')

        wellx2 = np.column_stack((dataDF['well'], dataDF['well']))
        startx2 = np.column_stack((dataDF['start'], dataDF['start']))
        stopx2 = np.column_stack((dataDF['stop'], dataDF['stop']))
        md = np.column_stack((dataDF['start'], dataDF['stop']))

        df = pd.DataFrame({'well': wellx2, 'md': md, 'start': startx2, 'stop': stopx2})
        return df
    @classmethod
    def extract_markers(cls, well, file_path = None):
        _file_path = file_path or Naming.default_marker_file('raw')
        xlsx_file = MemoryCache.get_instance().get(_file_path)
        if xlsx_file is None:
            xlsx_file = pd.ExcelFile(Naming.data_path(_file_path), engine='openpyxl')
            MemoryCache.get_instance().put(_file_path, xlsx_file)
        allMarkerDF = xlsx_file.parse(0, header=0)
        columns = list(allMarkerDF.columns)
        firstColumn = columns[0]
        allMarkerDF[firstColumn] = allMarkerDF[firstColumn].astype(str)
        filteredDF = allMarkerDF[allMarkerDF[firstColumn] == well]
        filteredDF.sort_values(by='MD', ascending=True, inplace=True)
        return filteredDF

    @classmethod
    def extract_zones(cls, well, file_path = None):
        markerDF = cls.extract_markers(well, file_path)
        columns = list(markerDF.columns)
        zoneDF = pd.DataFrame()
        zoneDF[columns[0]] = markerDF[columns[0]]
        zoneDF[columns[1]] = markerDF[columns[1]]
        zoneDF['start'] = markerDF[columns[5]].astype(float)
        zoneDF['stop'] = markerDF[columns[5]].shift(periods=-1).astype(float)
        print(zoneDF)
        return zoneDF

    @classmethod
    def extract_zones1(cls, well, file_path = None):
        _well = well
        if "-" in well:
            _well = re.sub(r'^.+\-', '', well)

        markerDF = cls.extract_markers(_well, file_path)
        columns = list(markerDF.columns)

        markerDF = markerDF.sort_values(columns[5])
        markerDF = markerDF.reset_index()
        if len(markerDF.index) < 2:
            return None, None
        max_depth = markerDF.iloc[-1][columns[5]]
        keyMarkerDF = markerDF[markerDF[columns[1]].str.startswith('SH')]

        keyZoneDF = pd.DataFrame()
        keyZoneDF[columns[0]] = keyMarkerDF[columns[0]]
        keyZoneDF[columns[1]] = keyMarkerDF[columns[1]]
        keyZoneDF[f"{columns[1]}-1"] = keyMarkerDF[columns[1]].shift(periods=-1)
        keyZoneDF['start'] = keyMarkerDF[columns[5]].astype(float)
        keyZoneDF['stop'] = keyMarkerDF[columns[5]].shift(periods=-1).astype(float)
        keyZoneDF.iat[-1, 4] = max_depth


        zoneDF = pd.DataFrame()
        zoneDF[columns[0]] = markerDF[columns[0]]
        zoneDF[columns[1]] = markerDF[columns[1]]
        zoneDF[f"{columns[1]}-1"] = markerDF[columns[1]].shift(periods=-1)
        zoneDF['start'] = markerDF[columns[5]].astype(float)
        zoneDF['stop'] = markerDF[columns[5]].shift(periods=-1).astype(float)

        zoneDF = zoneDF[( zoneDF[columns[1]].str.startswith('T-') & zoneDF[ f"{columns[1]}-1" ].str.startswith('T-') ) |
                        ( zoneDF[columns[1]].str.startswith('T-') & zoneDF[ f"{columns[1]}-1" ].str.startswith('B-') ) |
                        ( zoneDF[columns[1]].str.startswith('T-') & zoneDF[ f"{columns[1]}-1" ].str.startswith('SH') ) |
                        ( zoneDF[columns[1]].str.startswith('SH') & zoneDF[ f"{columns[1]}-1" ].str.startswith('B-') ) ]
        
        print(keyZoneDF)                
        print(zoneDF)
        return keyZoneDF, zoneDF
    @classmethod
    def extract_layers(cls, file_path=None):
        _file_path = file_path or Naming.default_perforation_file(category='raw')
        print(_file_path)
        xlsx_file = MemoryCache.get_instance().get(_file_path)
        if xlsx_file is None:
            xlsx_file = pd.ExcelFile(Naming.data_path(_file_path), engine='openpyxl')
            MemoryCache.get_instance().put(_file_path, xlsx_file)
        df = xlsx_file.parse(0, skiprows=3, header=None)
        layers = df.iloc[:, 5].unique()
        print(type(layers), layers)
        return layers
    @classmethod
    def extract_production_data(cls, wells=None, file_path=None, sheet=4):
        _file_path = file_path or Naming.default_production_monthly_file(category='raw')
        xlsx_file = MemoryCache.get_instance().get(_file_path)
        if xlsx_file is None:
            xlsx_file = pd.ExcelFile(Naming.data_path(_file_path), engine='openpyxl')
            MemoryCache.get_instance().put(_file_path, xlsx_file)
        df = xlsx_file.parse(sheet, header=0)
        retrieved_cols = {
            "Date": 0, 
            "Well": 1, 
            "Platform": 4, 
            "CV.OilRate": 6, 
            "Qoil/1000": 7, 
            "CV.LiquidRate": 9, 
            "Qwater/1000": 10, 
            "WaterProdCum": 11, 
            "CV.WaterInj_Rate": 14, 
            "CV.WaterCut": 17
        }
        ori_cols = [str(df.columns[c]) for c in retrieved_cols.values()]
        print(ori_cols)
        new_cols = list(retrieved_cols.keys())
        col_mapping = { c: new_cols[idx] for idx,c in enumerate(ori_cols) }
        print(col_mapping)
        
        df = df[ori_cols]
        df = df.rename( columns=col_mapping )
        df["Well"] = df["Well"].astype(str)
        print(df)
        print(wells)
        if wells and len(wells) > 0:
            df = df[df["Well"].isin(wells)]
        return df

    @classmethod
    def save_dataframe(cls, df, dest):
        df.to_excel(dest, index=False)
