import os
import json
import lasio
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from config.settings import DataConfig
import pandas as pd

def create_excel_tools(mcp_server, data_config: DataConfig) -> List[str]:
    _cACHE = dict()
    @mcp_server.tool(
        name="show_sheets",
        description="show sheets in an excel file"
    )
    def show_sheets(file_path: str = None, **kwargs):
        try:
            ori_file_path = kwargs['input']
            file_path = os.path.join(data_config.data_dir, ori_file_path)
            print(f"file_path = {file_path}");
            if not file_path:
                return { "text": "file to plot is empty or None" }
            elif not os.path.exists(file_path):
                return { "text": f"{file_path} does not exist" }
            elif not os.path.isfile(file_path):
                return { "text": f"{file_path} is not a regular file" }

            excel_file = None
            if file_path in _cACHE:
                excel_file = _cACHE[file_path]
            else:
                excel_file = pd.ExcelFile(file_path, engine='openpyxl')

            return {"text": json.dumps(excel_file.sheet_names)}
        except Exception as e:
            traceback.print_exc()
            return {"text": "Tool failed: {str(e)}"}

    @mcp_server.tool(
        name="show_columns",
        description="show columns in a sheet of an excel file"
    )
    def show_columns(**kwargs):
        try:
            input_data = json.loads(kwargs['input'])
            ori_file_path = input_data['file_path']
            sheet = input_data.get('sheet', 0)
            file_path = os.path.join(data_config.data_dir, ori_file_path)
            print(f"file_path = {file_path}, sheet={sheet}");
            if not file_path:
                return { "text": "file to plot is empty or None" }
            elif not os.path.exists(file_path):
                return { "text": f"{file_path} does not exist" }
            elif not os.path.isfile(file_path):
                return { "text": f"{file_path} is not a regular file" }

            excel_file = None
            if file_path in _cACHE:
                excel_file = _cACHE[file_path]
            else:
                excel_file = pd.ExcelFile(file_path, engine='openpyxl')

            sheetDF = excel_file.parse(sheet, header=1)
            print(sheetDF.columns)

            return {"text": json.dumps(list(sheetDF.columns))}
        except Exception as e:
            traceback.print_exc()
            return {"text": "Tool failed: {str(e)}"}
    tool_names = [
        "show_sheets",
        "show_columns"
    ]

    return tool_names

