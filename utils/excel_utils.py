from cache import MemoryCache
import pandas as pd


def parse_excel(file_path: str, sheet: int = 0):
    excel_file = MemoryCache.get_instance().get(file_path)
    if excel_file is None:
        excel_file = pd.ExcelFile(file_path, engine="openpyxl")
        MemoryCache.get_instance().put(file_path, excel_file)
    df = excel_file.parse(
        sheet,
        header=0,
    )
    return df


PRODUCTION_FILEPATH = "data/production/PVT_WellTest_Perforation_WaterAnalysis.xlsx"
PROD_WELL_COL = 1


def parse_well_production(file_path: str = PRODUCTION_FILEPATH, sheet: int = 4):
    df = parse_excel(file_path, sheet)
    # convert well number to string
    df[df.columns[PROD_WELL_COL]] = df[df.columns[PROD_WELL_COL]].astype(str)
    return df


MARKER_WELL_COL = 0


def parse_marker(file_path: str = "data/misc/Marker.xlsx", sheet: int = 0):
    df = parse_excel(file_path, sheet)
    # convert well number to string
    df[df.columns[MARKER_WELL_COL]] = df[df.columns[MARKER_WELL_COL]].astype(str)
    return df
