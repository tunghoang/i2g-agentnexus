import os
import numpy as np
import json
from glob import glob, iglob
import traceback
from datetime import datetime
from typing import List, Dict, Any
from config.settings import DataConfig
from store import Store
from naming import Naming
from cache import MemoryCache
import pandas as pd
import lasio
from calendar import monthrange
from pywaterflood import CRM
from utils.excel_utils import PROD_WELL_COL, parse_well_production
from utils.missing_pay_utils import well_checklist
from utils.plot_utils import multi_chart, advLogplot, logplot


def create_missingpay_tools(mcp_server, data_config: DataConfig) -> List[str]:
    WELLS_DIR_PATH = "wells"

    @mcp_server.tool(
        name="build_logplot", description="get marker for well from marker file"
    )
    def build_logplot(input):
        try:
            input_data = json.loads(input)
            well = input_data.get("well")
            if well is None:
                raise Exception("Please specify well to plot")
            track_templates = input_data.get("track_templates", None)
            if track_templates is None:
                raise Exception("Track templates should be specified")

            well_data_dir = f'wells/{well}'
            if not os.path.isdir(f"{data_config.data_dir}/{well_data_dir}"):
                raise Exception(f"Well {well} not found")
            composite_logs = f"{well_data_dir}/GIS/Las/*.las"
            files = glob(f"{data_config.data_dir}/{composite_logs}")
            if len(files) == 0:
                raise Exception(f'No composite logs found for well {well}')

            track_templates = input_data.get('track_templates', 'GR,LLD,NPHI')
            track_templates = track_templates.split(',')
            track_templates = [ tpl.strip() for tpl in track_templates ]

            logDF_path = files[0]
            
            las = MemoryCache.get_instance().get(logDF_path)
            if las is None: 
                las = lasio.read(logDF_path)
                MemoryCache.get_instance().put(logDF_path, las)

            df = las.df().reset_index()

            
            if track_templates:
                fig = advLogplot(df, las.curves, track_styles=track_templates, title=f"Well {well} Logplot")
            else:
                fig = logplot(df, las.curves)
            
            dest_path = Naming.dest_path(logDF_path.removeprefix(f"{data_config.data_dir}/"), category='logplot')
            fig.write_html(dest_path)
            return {"text": Naming.publish_path(logDF_path.removeprefix(f"{data_config.data_dir}/"), category='logplot')}
        except Exception as e:
            traceback.print_exc()
            return {"text": str(e)}

    @mcp_server.tool(
        name="well_checklist_table", description="get checklist table of well logs data"
    )
    def well_checklist_table(**kwargs):
        try:
            input_data = json.loads(kwargs["input"])
            well_names_input: list[str] = input_data.get("wells")
            wells_dir = os.path.join(data_config.data_dir, WELLS_DIR_PATH)
            (
                well_names,
                loai_gieng_result,
                ten_gian_result,
                log_result,
                devi_result,
                mudlog_result,
                marker_result,
                thu_via_result,
                plt_result,
                kqdvl_result,
                log_details,
            ) = well_checklist(wells=well_names_input)
            count = len(well_names)
            out_file_relative_path = os.path.join(
                f"well_checklist_table{"_".join(well_names)}.html"
            )
            out_file_path = os.path.join("/tmp", out_file_relative_path)
            df = pd.DataFrame(
                data={
                    "STT": range(1, count + 1),
                    "Tên giếng": well_names,
                    "Thân": ["N/A"] * count,
                    "Loại giếng": loai_gieng_result,
                    "Tên giàn": ten_gian_result,
                    "KB (m)": ["N/A"] * count,
                    "Năm khoan": ["N/A"] * count,
                    "Móng": ["N/A"] * count,
                    "Đáy giếng MD": ["N/A"] * count,
                    "Đáy giếng TVDss": ["N/A"] * count,
                    "Đối tượng khoan": ["N/A"] * count,
                    "Log": log_result,
                    "Deviation": devi_result,
                    "Mudlog": mudlog_result,
                    "Marker": marker_result,
                    "Thử vỉa": thu_via_result,
                    "PLT": plt_result,
                    "KQĐVL": kqdvl_result,
                }
            )

            table = df.to_html(index=False, justify="center")
            template = open("templates/well_checklist_tpl.html", "r").read()
            result = template.replace("{{TABLE}}", table)
            with open(out_file_path, "w") as f:
                f.write(result)

            return {"text": out_file_relative_path}
        except Exception as e:
            traceback.print_exc()
            return {"text": f"Tool failed: {str(e)}"}

    tool_names = [
        "build_logplot",
        "well_checklist_table",
    ]
    return tool_names
