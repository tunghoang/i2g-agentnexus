import os
import numpy as np
import json
from glob import glob, iglob
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
from utils.plot_utils import multi_chart, advLogplot, logplot

def create_missingpay_tools(mcp_server, data_config: DataConfig) -> List[str]:
    @mcp_server.tool(
        name="build_logplot",
        description="get marker for well from marker file"
    )
    def build_logplot(input):
        try:
            input_data = json.loads(input)
            well = input_data.get('well')
            if well is None:
                raise Exception("Please specify well to plot")
            track_templates = input_data.get('track_templates', None)
            if track_templates is None:
                raise Exception("Track templates should be specified")

            well_data_dir = 'wells/{well}'
            if not os.path.isdir(well_data_dir):
                raise Exception(f"Well {well} not found")
            composite_logs = f"{well_data_dir}/GIS/Las/*.las"
            files = glob(composite_logs)
            if len(files) == 0:
                raise Exception(f'No composite logs found for well {well}')
            return {"text": "Build success"}

        except Exception as e:
            traceback.print_exc()
            return {"text": str(e), "isError": true}
            
    tool_names = [
        "build_logplot"
    ]
    return tool_names
