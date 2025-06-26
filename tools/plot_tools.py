import os
import json
import lasio
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from config.settings import DataConfig

import plotly.graph_objects as go
from plotly.subplots import make_subplots
__COLOR_PALETTE = ('#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000')

def getLineConfig(curveName):
    if curveName in ['VCLAV']:
        return dict(color=__COLOR_PALETTE[0], dash='solid')
    elif curveName in ['PHIE']:
        return dict(color=__COLOR_PALETTE[1], dash='solid')
    elif curveName in ['SW']:
        return dict(color=__COLOR_PALETTE[2], dash='solid')
    elif curveName in ['RESFLAG', 'NETFLAG']:
        return dict(color=__COLOR_PALETTE[-1], dash='solid')
    else:
        return dict(color='black', dash='solid')

def create_plot_tools(mcp_server, data_config: DataConfig) -> List[str]:
    CHART_DIR = '/tmp'
    @mcp_server.tool(
        name="plot_las",
        description="Plot a las file in a plotly chart"
    )
    def plot_las(file_path: str = None, **kwargs):
        try:
            ori_file_path = kwargs['input']
            dest_path = f'{CHART_DIR}/{ori_file_path}.html'
            containing_dir = os.path.join(CHART_DIR, os.path.dirname(ori_file_path))
            file_path = os.path.join(data_config.data_dir, ori_file_path)
            if not file_path:
                return { "text": "file to plot is empty or None" }
            elif not os.path.exists(file_path):
                return { "text": f"{file_path} does not exist" }
            elif not os.path.isfile(file_path):
                return { "text": f"{file_path} is not a regular file" }

            if os.path.exists(dest_path) and (datetime.now().timestamp() - os.path.getmtime(dest_path)) < 3 * 60:
                return {"text": f'{ori_file_path}.html'}

            print(containing_dir, dest_path)          
            Path(containing_dir).mkdir(parents=True, exist_ok=True)
            las = lasio.read(file_path)
            df = las.df().reset_index()
            curveNames = df.columns[1:]
            refCurveName = df.columns[0]
            fig = make_subplots(rows=1, cols=len(curveNames), shared_yaxes=True, horizontal_spacing=0)
            for idx, c in enumerate(curveNames):
                trace = go.Scatter(x=df[c], y = df[refCurveName], name=c, line=getLineConfig(c))
                fig.append_trace(trace, 1, idx + 1)

            columnWidth = 90
            fig.update_xaxes(
                showline=True,
                linewidth=0.5,
                linecolor='#444',
                mirror=True,
                showticklabels=True,
                side='top',
                #rangeslider=dict(visible=True, borderwidth=1, thickness=0.05),
                gridcolor='#eee',
                position=1)
            fig.update_yaxes(showline=True, linewidth=0.5, linecolor='#444', mirror=True, autorange='reversed', gridcolor='#eee')
            fig.update_layout(
                plot_bgcolor='#fff',
                overwrite=True,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                width=columnWidth * len(curveNames),
                height= 500,
                autosize=False)
            html_code = fig.write_html(dest_path)
            return {"text": f'{ori_file_path}.html'}
        except Exception as e:
            traceback.print_exc()
            return {"text": "Ploting las failed: {str(e)}"}
    tool_names = [
        "plot_las"
    ]

    return tool_names


