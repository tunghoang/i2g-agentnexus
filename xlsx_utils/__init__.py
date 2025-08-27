import pandas as pd
import numpy as np
import re
from cache import MemoryCache
from naming import Naming

class XLSX:
    #PRODUCTION_MONTHLY_SHEET=4
    PRODUCTION_MONTHLY_SHEET=0

    @classmethod
    def parse_excel(cls, file_path: str, sheet: int = 0, header:int = 0,skiprows=None):
        excel_file = MemoryCache.get_instance().get(file_path)
        if excel_file is None:
            excel_file = pd.ExcelFile(file_path, engine="openpyxl")
            MemoryCache.get_instance().put(file_path, excel_file)
        df = excel_file.parse(
            sheet,
            header=header,skiprows=skiprows
        )
        return df

    @classmethod
    def parse_well_production(cls, sheet: int = PRODUCTION_MONTHLY_SHEET):
        PROD_WELL_COL = 1
        df = cls.parse_excel(Naming.default_production_monthly_file(category='store'), sheet)
        #df['Date'] = pd.to_datetime(df['Date'], unit='D', origin='1899-12-30')
        df['Date'] = pd.to_datetime(df['Date'])
        # convert well number to string
        df[df.columns[PROD_WELL_COL]] = df[df.columns[PROD_WELL_COL]].astype(str)
        return df

    @classmethod
    #def parse_plt(cls, sheet='PLT'):
    def parse_plt(cls, sheet=0):
        df = cls.parse_excel(Naming.default_plt_file(category='store'), sheet)
        return df
    @classmethod
    def parse_water_analysis(cls, sheet: int = 0):
        df = cls.parse_excel(Naming.default_water_analysis_file(category='store'), sheet=sheet, header=1)
        columns = list(df.columns)
        df = df.rename(columns={columns[0]:'Well', columns[3]: 'Date', columns[4]: '%WaterProd', columns[5]: '%WaterInj'})
        df['Well'] = df['Well'].astype(str)
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
        PERFORATION_SHEET = "BH"
        well_number = well
        tokens = well.split("-")
        if len(tokens) > 1:
            well_number = tokens[1]
            PERFORATION_SHEET = tokens[0]
        
        WELL_COLUMN = 2
        MD_TOP = 8
        MD_BOTTOM = 9
        perforation_file = Naming.default_perforation_file(category='raw')
        xlsx_file = MemoryCache.get_instance().get(perforation_file)
        if xlsx_file is None:
           xlsx_file = pd.ExcelFile(Naming.data_path(perforation_file), engine='openpyxl')
           MemoryCache.get_instance().put(perforation_file, xlsx_file)

        sheetDF = xlsx_file.parse(PERFORATION_SHEET, skiprows=3, header=None)
        dataDF = sheetDF.iloc[:, [WELL_COLUMN, MD_TOP, MD_BOTTOM]]
        dataDF = dataDF.rename(columns={
            dataDF.columns[0]: "well", 
            dataDF.columns[1]: 'start',
            dataDF.columns[2]: 'stop'
        })
        dataDF['well'] = dataDF['well'].astype(str)
        dataDF = dataDF[dataDF.well == well_number]
        if dataDF.empty:
            raise Exception(f'Well {well} does not exist in perforation file')
        dataDF = dataDF.sort_values(by="start", ascending=True)

        # merge adjacent zones
        rows = []
        prev_start = None
        prev_stop = None
        for idx,row in dataDF.iterrows():
            if row['start'] != prev_stop:
                rows.append(row)
                prev_stop = row['stop']
            else:
                rows[-1]['stop'] = row['stop']

        dataDF = pd.DataFrame(rows)

        wellx2 = np.column_stack((dataDF['well'], dataDF['well'])).flatten()
        startx2 = np.column_stack((dataDF['start'], dataDF['start'])).flatten()
        stopx2 = np.column_stack((dataDF['stop'], dataDF['stop'])).flatten()
        md = np.column_stack((dataDF['start'], dataDF['stop'])).flatten()

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
    def extract_production_data(cls, 
        wells=None, file_path=None, sheet=PRODUCTION_MONTHLY_SHEET,
        idxcols = [0,1,4,6,7,9,10,11,14,17],
        colnames = ['Date', 'Well', 'Platform', 'CV.OilRate', 'Qoil/1000', 'CV.LiquidRate', 'Qwater/1000', 'WaterProdCum', 'CV.WaterInj_Rate', 'CV.WaterCut']
    ):
        _file_path = file_path or Naming.default_production_monthly_file(category='raw')
        xlsx_file = MemoryCache.get_instance().get(_file_path)
        if xlsx_file is None:
            xlsx_file = pd.ExcelFile(Naming.data_path(_file_path), engine='openpyxl')
            MemoryCache.get_instance().put(_file_path, xlsx_file)
        df = xlsx_file.parse(sheet, header=0)
        #retrieved_cols = {
        #    "Date": 0, 
        #    "Well": 1, 
        #    "Platform": 4, 
        #    "CV.OilRate": 6, 
        #    "Qoil/1000": 7, 
        #    "CV.LiquidRate": 9, 
        #    "Qwater/1000": 10, 
        #    "WaterProdCum": 11, 
        #    "CV.WaterInj_Rate": 14, 
        #    "CV.WaterCut": 17
        #}
        if colnames and len(colnames):
            retrieved_cols = { colnames[idx]: idxcols[idx] for idx,_ in enumerate(idxcols) }
            ori_cols = [str(df.columns[c]) for c in retrieved_cols.values()]
            print(ori_cols)
            new_cols = list(retrieved_cols.keys())
            col_mapping = { c: new_cols[idx] for idx,c in enumerate(ori_cols) }
            print(col_mapping)
            
            df = df[ori_cols]
            df = df.rename( columns=col_mapping )
        else:
            df = df[[str(df.columns[c]) for c in idxcols]]
        df[df.columns[1]] = df[df.columns[1]].astype(str)
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]]).astype(str)
        print(df)
        print(wells)
        if wells and len(wells) > 0:
            df = df[df[df.columns[1]].isin(wells)]
        return df

    @classmethod
    def extract_welltest(cls, wells:list[str]):
        _filekey = Naming.default_welltest_file(category="raw")
        _filepath = Naming.default_welltest_file(category="store")

        PRO_SHEET = "Mio_production"        
        INJ_SHEET = "Mio_injection"        

        xlsx_file = MemoryCache.get_instance().get(_filekey)
        if xlsx_file is None:
            xlsx_file = pd.ExcelFile(_filepath, engine='openpyxl')
            MemoryCache.get_instance().put(_filekey, xlsx_file)
        pro_df = xlsx_file.parse(PRO_SHEET, header=0)
        pro_df = pro_df.rename(columns={pro_df.columns[0]: "Date"})
        pro_df.columns = pro_df.columns.astype(str)
        print('production_df 1', pro_df)
        print(pro_df.columns, wells)
        pro_df = pro_df[ ['Date'] + [ w for w in wells if w in list(pro_df.columns) ] ]
        pro_df = pro_df.set_index('Date')
        pro_df = pro_df.dropna(how='all').reset_index()
        print('production_df', pro_df)

        inj_df = xlsx_file.parse(INJ_SHEET, header=0)
        inj_df = inj_df.rename(columns={inj_df.columns[0]: "Date"})
        inj_df.columns = inj_df.columns.astype(str)
        inj_df = inj_df[ ['Date'] + [ w for w in wells if w in list(inj_df.columns) ] ]
        inj_df = inj_df.set_index('Date')
        inj_df = inj_df.dropna(how='all').reset_index()
        print(inj_df)
        return {"production": pro_df, "injection": inj_df}

    @classmethod
    def save_dataframe(cls, df, dest):
        df.to_excel(dest, index=False)

    @classmethod
    def write_excel(cls, dfs: dict[pd.DataFrame], filepath:str, index=True):
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for key,df in dfs.items():
                df.to_excel(writer, sheet_name=key, index=index)

    @classmethod
    def extract_wellpos(cls, wells: list[str]):
        file_path = Naming.default_wellpos_file(category='store')
        df = MemoryCache.get_instance().get(file_path)
        if df is None:
            df = pd.read_csv(file_path, header=0, sep=" ")
            df['Well1'] = 'BH-' + df['Well'].astype(str)
            MemoryCache.get_instance().put(file_path, df)
        if wells and len(wells) > 0:
            return df[df['Well'].isin(wells) | df['Well1'].isin(wells)]
        return df
