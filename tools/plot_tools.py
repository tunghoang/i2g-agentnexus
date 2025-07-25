import os
import json
import time
import lasio
import traceback
import hashlib
from glob import glob
from cache import MemoryCache
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, TypedDict

from python_a2a import FastMCP
from config.settings import DataConfig

from utils.plot_utils import logplot, advLogplot, histogram
from naming import Naming

import plotly.graph_objects as go
from plotly.subplots import make_subplots

__COLOR_PALETTE = (
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcd60c",
    "#008080",
    "#9a6324",
    "#800000",
    "#808000",
    "#000075",
    "#808080",
    "#000000",
)


def getHashColor(curveName):
    encoded_string = curveName.encode("utf-8")
    hash_object = hashlib.sha256(encoded_string)
    hashed_integer = int(hash_object.hexdigest(), 16)
    idx = hashed_integer % len(__COLOR_PALETTE)
    return __COLOR_PALETTE[idx]


def getLineConfig(curveName):
    return dict(color=getHashColor(curveName), dash="solid")
    if curveName in ["VCLAV"]:
        return dict(color=__COLOR_PALETTE[0], dash="solid")
    elif curveName in ["PHIE"]:
        return dict(color=__COLOR_PALETTE[1], dash="solid")
    elif curveName in ["SW"]:
        return dict(color=__COLOR_PALETTE[2], dash="solid")
    elif curveName in ["RESFLAG", "NETFLAG"]:
        return dict(color=__COLOR_PALETTE[-1], dash="solid")
    else:
        return dict(color="black", dash="solid")


def getColor(curveName):
    return getLineConfig(curveName)["color"]


def borderColor():
    return "#444"


def headerFillColor():
    return "#eee"


def create_plot_tools(mcp_server: FastMCP, data_config: DataConfig) -> List[str]:
    CHART_DIR = "/tmp"
    HEADER_HEIGHT = 90
    columnWidth = 150
    _LAS_CACHE = dict()

    @mcp_server.tool(name="plot_las", description="Plot a las file in a plotly chart")
    def plot_las(**kwargs):
        try:
            input_data = json.loads(kwargs['input'])
            #ori_file_path = kwargs["input"]
            ori_file_path = input_data['file_path']
            track_templates = input_data.get('templates', None)
            if track_templates:
                track_templates = track_templates.split(',')
                track_templates = [tpl.strip() for tpl in track_templates]

            print(track_templates)
            dest_path = f"{CHART_DIR}/{ori_file_path}.html"
            containing_dir = os.path.join(CHART_DIR, os.path.dirname(ori_file_path))
            file_path = os.path.join(data_config.data_dir, ori_file_path)
            print(ori_file_path, file_path)
            if not file_path:
                return {"text": "file to plot is empty or None"}
            elif not os.path.exists(file_path):
                return {"text": f"{file_path} does not exist"}
            elif not os.path.isfile(file_path):
                return {"text": f"{file_path} is not a regular file"}

            if (
                os.path.exists(dest_path)
                and (datetime.now().timestamp() - os.path.getmtime(dest_path)) < 3 * 60
            ):
                return {"text": f"{ori_file_path}.html"}

            print(containing_dir, dest_path)
            Path(containing_dir).mkdir(parents=True, exist_ok=True)
            las = (
                lasio.read(file_path)
                if not file_path in _LAS_CACHE
                else _LAS_CACHE[file_path]
            )
            df = las.df().reset_index()
            if track_templates:
                fig = advLogplot(df, las.curves, track_styles=track_templates)
            else:
                fig = logplot(df, las.curves)
            html_code = fig.write_html(dest_path)
            return {"text": f"{ori_file_path}.html"}
        except Exception as e:
            traceback.print_exc()
            return {"text": f"Ploting las failed: {str(e)}", "isError": True}
    @mcp_server.tool(name="plot_logplot", description="Plot a las file in a plotly chart with spec")
    def plot_logplot(**kwargs):
        try:
            ori_file_path = kwargs["input"]
            dest_path = f"{CHART_DIR}/{ori_file_path}.html"
            containing_dir = os.path.join(CHART_DIR, os.path.dirname(ori_file_path))
            file_path = os.path.join(data_config.data_dir, ori_file_path)
            print(ori_file_path, file_path)
            if not file_path:
                return {"text": "file to plot is empty or None"}
            elif not os.path.exists(file_path):
                return {"text": f"{file_path} does not exist"}
            elif not os.path.isfile(file_path):
                return {"text": f"{file_path} is not a regular file"}

            if (
                os.path.exists(dest_path)
                and (datetime.now().timestamp() - os.path.getmtime(dest_path)) < 3 * 60
            ):
                return {"text": f"{ori_file_path}.html"}

            print(containing_dir, dest_path)
            Path(containing_dir).mkdir(parents=True, exist_ok=True)
            las = (
                lasio.read(file_path)
                if not file_path in _LAS_CACHE
                else _LAS_CACHE[file_path]
            )
            df = las.df().reset_index()
            fig = advLogplot(df, las.curves, track_styles=["INTERP", "FLAGS"], title="My Log plot")
            html_code = fig.write_html(dest_path)
            return {"text": f"{ori_file_path}.html"}
        except Exception as e:
            traceback.print_exc()
            return {"text": f"Ploting las failed: {str(e)}", "isError": True}

    @mcp_server.tool(
        name="plot_histogram_las",
        description="Plot histogram of a las file in a plotly chart",
    )
    def plot_histogram_las(input: str):
        try:
            input_data = json.loads(input)
            ori_file_path = input_data["file_path"]
            out_file_path = os.path.join("histogram", ori_file_path)
            dest_path = f"{CHART_DIR}/{out_file_path}.html"
            containing_dir = os.path.join(CHART_DIR, os.path.dirname(out_file_path))
            file_path = os.path.join(data_config.data_dir, ori_file_path)
            if not file_path:
                return {"text": "file to plot is empty or None"}
            elif not os.path.exists(file_path):
                return {"text": f"{file_path} does not exist"}
            elif not os.path.isfile(file_path):
                return {"text": f"{file_path} is not a regular file"}

            # if (
            #     os.path.exists(dest_path)
            #     and (datetime.now().timestamp() - os.path.getmtime(dest_path)) < 3 * 60
            # ):
            #     return {"text": f"{out_file_path}.html"}

            Path(containing_dir).mkdir(parents=True, exist_ok=True)
            las = (
                lasio.read(file_path)
                if not file_path in _LAS_CACHE
                else _LAS_CACHE[file_path]
            )
            df = las.df().reset_index()
            curve_names = input_data["curve_names"]
            num_bins = input_data.get("num_bins", 10)
            fig = histogram(df, curve_names, num_bins, file_path=file_path)
            fig.write_html(dest_path)
            return {"text": f"{out_file_path}.html"}
        except Exception as e:
            traceback.print_exc()
            return {"text": "Ploting histogram las failed: {str(e)}"}

    @mcp_server.tool(
        name="plot_histogram_well",
        description="Plot histogram of a well in a plotly chart",
    )
    def plot_histogram_well(input: str):
        try:
            input_data = json.loads(input)
            well = input_data["well"]
            if well is None:
                raise Exception("Please specify well to plot")

            well_data_dir = f'wells/{well}'
            if not os.path.isdir(f"{data_config.data_dir}/{well_data_dir}"):
                raise Exception(f"Well {well} not found")
            composite_logs = f"{well_data_dir}/GIS/Las/*.las"
            files = glob(f"{data_config.data_dir}/{composite_logs}")
            if len(files) == 0:
                raise Exception(f'No composite logs found for well {well}')

            curves = input_data.get("curves", None)
            if curves is None:
                raise Exception("Track curves should be specified")

            curves = curves.split(',')
            curves = [ c.strip() for c in curves ]

            logDF_path = files[0]
            
            las = MemoryCache.get_instance().get(logDF_path)
            if las is None: 
                las = lasio.read(logDF_path)
                MemoryCache.get_instance().put(logDF_path, las)

            df = las.df().reset_index()

            num_bins = input_data.get("num_bins", 10)
            if len(set(curves).intersection(set(list(df.columns))) ) == 0:
                raise Exception(f"{curves} not found")

            fig = histogram(df, curves, num_bins, file_path=well)
            dest_path = Naming.dest_path(logDF_path.removeprefix(f"{data_config.data_dir}/"), category='histogram')
            fig.write_html(dest_path)
            return {"text": Naming.publish_path(logDF_path.removeprefix(f"{data_config.data_dir}/"), category='histogram')}
        except Exception as e:
            traceback.print_exc()
            return {"text": "Ploting histogram las failed: {str(e)}"}

    tool_names = ["plot_las", "plot_histogram_las", "plot_histogram_well"]

    return tool_names
