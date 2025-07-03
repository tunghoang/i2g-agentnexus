import os
import hashlib
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

def logplot(df, curves):
    HEADER_HEIGHT = 90
    columnWidth = 150

    curveNames = df.columns[1:]
    refCurveName = df.columns[0]
    fig = make_subplots(
        rows=1, cols=len(curveNames), shared_yaxes=True, horizontal_spacing=0
    )
    for idx, c in enumerate(curveNames):
        trace = go.Scatter(
            x=df[c], y=df[refCurveName], name=c, line=getLineConfig(c)
        )
        fig.append_trace(trace, 1, idx + 1)
        fig.update_xaxes(
            nticks=4, tickfont_color=getColor(c), row=1, col=idx + 1
        )
        fig.add_shape(
            type="rect",
            x0=0,
            x1=1,
            xref="x domain",
            y0=2,
            y1=HEADER_HEIGHT * 2.0 / 3.0,
            yref="y domain",
            ysizemode="pixel",
            yanchor=1,
            layer="below",
            visible=True,
            line_width=1,
            line_color=borderColor(),
            fillcolor=headerFillColor(),
            row=1,
            col=idx + 1,
        )
        fig.add_hline(
            y=HEADER_HEIGHT / 3,
            yref="y domain",
            ysizemode="pixel",
            yanchor=1,
            layer="below",
            visible=True,
            line_width=1,
            line_color=getColor(c),
            row=1,
            col=idx + 1,
        )
        fig.add_annotation(
            text=f"{c}({curves[c].unit})",
            font=dict(color="white", size=10),
            bgcolor=getColor(c),
            showarrow=False,
            x=0.5,
            y=1,
            xanchor="center",
            yanchor="middle",
            xref="x domain",
            yref="y domain",
            yshift=HEADER_HEIGHT / 2,
            align="center",
            visible=True,
            row=1,
            col=idx + 1,
        )

    fig.update_xaxes(
        showline=True,
        linewidth=0.5,
        linecolor="#444",
        mirror=True,
        showticklabels=True,
        side="top",
        # rangeslider=dict(visible=True, borderwidth=1, thickness=0.05),
        gridcolor="#eee",
        position=1,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=0.5,
        linecolor="#444",
        mirror=True,
        autorange="reversed",
        gridcolor="#eee",
    )
    fig.update_layout(
        plot_bgcolor="#fff",
        overwrite=True,
        showlegend=False,
        margin=dict(l=0, r=0, t=HEADER_HEIGHT, b=0),
        width=columnWidth * len(curveNames),
        height=500,
        autosize=False,
    )
    return fig

def histogram(df, curve_names, num_bins, file_path: str=""):
    fig = go.Figure(
        data=[],
        layout=go.Layout(
            title=go.layout.Title(
                text=f"Histogram of {', '.join(curve_names)} from {os.path.basename(file_path)}",
                xref="paper",
                x=0,
            ),
            barmode="stack",
            width=1000,
            height=500,
        ),
    )
    for c in curve_names:
        trace = go.Histogram(x=df[c.strip()], nbinsx=num_bins, name=c.strip())
        fig.add_trace(trace)
    return fig

def multi_chart(chart_titles, data1, data2, data3):
    fig = make_subplots(
        cols=1, rows=len(chart_titles), shared_xaxes=True,
        subplot_titles=chart_titles
    )
    for i,title in enumerate(chart_titles):
        trace1 = go.Scatter(x=data1['x'], y=data1['y'][:, i], 
                            name=f'[{title}] Production', mode='lines', line=dict(color='blue'), legendgroup=f'{i}')
        trace2 = go.Scatter(x=data2['x'], y=data2['y'][:, i], 
                            name=f'[{title}] Train', mode='lines', line=dict(color='red'), legendgroup=f'{i}')
        trace3 = go.Scatter(x=data3['x'], y=data3['y'][:, i], 
                            name=f'[{title}] Predict', mode='lines', line=dict(color='green'), legendgroup=f'{i}')

        fig.append_trace(trace1, i+1, 1)
        fig.append_trace(trace2, i+1, 1)
        fig.append_trace(trace3, i+1, 1)

    fig.update_xaxes(
        showline=True,
        linewidth=0.5,
        linecolor="#444",
        showticklabels=True,
        gridcolor="#eee",
    )
    fig.update_yaxes(
        showline=True,
        linewidth=0.5,
        linecolor="#444",
        gridcolor="#eee",
    )
    fig.update_layout(
        plot_bgcolor="#fff",
        width=800,
        legend_tracegroupgap=100
    )
    return fig

