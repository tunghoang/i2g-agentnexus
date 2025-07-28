import os
import numpy as np
import hashlib
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_allTrackConfigs = None

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

def getTrackConfig(trackstyle):
    global _allTrackConfigs
    if _allTrackConfigs is None:
        with open('utils/track_template.yaml') as file:
            _allTrackConfigs = yaml.safe_load(file)
    return _allTrackConfigs.get(trackstyle)

def __depth_track():
    pass

def __track_header(fig, TRACK_HEADER, yanchor=1, TRACK_TITLE=20, xdomain='', colIdx = None, curve=None, unit=None):
    if xdomain:
        fig.add_shape(
            type="rect",
            x0=0,
            x1=1,
            xref=f"x{xdomain} domain",
            y0=2,
            y1=TRACK_HEADER,
            yref="y domain",
            ysizemode="pixel",
            yanchor=yanchor,
            layer="below",
            visible=True,
            line_width=0.5,
            line_color=borderColor(),
            fillcolor=headerFillColor(),
            #row=1,
            #col=colIdx + 1,
        )
        fig.add_shape(
            type="rect",
            xref=f'x{xdomain} domain', x0=0, x1=1,
            yref='y domain', ysizemode='pixel', yanchor=yanchor,y0=TRACK_HEADER,y1=TRACK_HEADER - TRACK_TITLE,
            visible=True, fillcolor='#f5deb3',
            line_width=0.5,
            line_color=borderColor(),
            #row=1,
            #col=colIdx + 1,
        )
        fig.add_annotation(
            text=f"({colIdx + 1})",
            font=dict(color="black", size=10),
            showarrow=False,
            align="center",
            visible=True,
            xref=f"x{xdomain} domain", xanchor="center", x = 0.5, 
            yref="y domain", yanchor="middle", y = 1,
            yshift=TRACK_HEADER - TRACK_TITLE/2,
            #row=1,
            #col=colIdx + 1,
        )
        if curve and unit:
            fig.add_annotation(text=f"{curve} ({unit})", 
                               font=dict(color="black", size=10),
                               showarrow=False,
                               align="center",
                               visible=True,
                               xref=f"x{xdomain} domain", xanchor="center", x = 0.5, 
                               yref=f"y domain", yanchor="middle", y = 1,
                               yshift=TRACK_HEADER - 3*TRACK_TITLE/2 - 6)
    else:
        fig.add_shape(
            type="rect",
            x0=0,
            x1=1,
            xref="x domain",
            y0=2,
            y1=TRACK_HEADER,
            yref="y domain",
            ysizemode="pixel",
            yanchor=yanchor,
            layer="below",
            visible=True,
            line_width=0.5,
            line_color=borderColor(),
            fillcolor=headerFillColor(),
            row=1,
            col=colIdx + 1,
        )
        fig.add_shape(
            type="rect",
            xref='x domain', x0=0, x1=1,
            yref='y domain', ysizemode='pixel', yanchor=yanchor,y0=TRACK_HEADER,y1=TRACK_HEADER - TRACK_TITLE,
            visible=True, fillcolor='#f5deb3',
            line_width=0.5,
            line_color=borderColor(),
            row=1,
            col=colIdx + 1,
        )
        fig.add_annotation(
            text=f"({colIdx + 1})",
            font=dict(color="black", size=10),
            showarrow=False,
            align="center",
            visible=True,
            xref="x domain", xanchor="center", x = 0.5, 
            yref="y domain", yanchor="middle", y = 1,
            yshift=TRACK_HEADER - TRACK_TITLE/2,
            row=1,
            col=colIdx + 1,
        )
        if curve and unit:
            fig.add_annotation(text=f"{curve} ({unit})", 
                               font=dict(color="black", size=10),
                               showarrow=False,
                               align="center",
                               visible=True,
                               xref="x domain", xanchor="center", x = 0.5, 
                               yref="y domain", yanchor="middle", y = 1,
                               yshift=TRACK_HEADER - 3*TRACK_TITLE/2 - 6,
                               row=1,
                               col=colIdx + 1)
    return fig

def __track_body(fig, TRACK_HEADER, trackbodyheight, xdomain='', colIdx=None ):
    if xdomain:
        fig.add_shape(
            type="rect",
            x0=0,
            x1=1,
            xref=f"x{xdomain} domain",
            y0=0,
            y1=1,
            yref="y domain",
            #layer="below",
            visible=True,
            line_width=0.5,
            line_color=borderColor(),
            #row=1,
            #col=colIdx + 1,
        )
    else:
        fig.add_shape(
            type="rect",
            x0=0,
            x1=1,
            xref="x domain",
            y0=0,
            y1=1,
            yref="y domain",
            #layer="below",
            visible=True,
            line_width=0.5,
            line_color=borderColor(),
            row=1,
            col=colIdx + 1,
        )
    if colIdx is not None and colIdx < 2:
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=colIdx + 1)
    return fig

def logplot(df, curves, title=None):
    PLOT_HEADER = 60
    TRACK_HEADER = 180
    PLOT_HEIGHT = 1000
    columnWidth = 150
    CURVE_HEADER = 30
    TRACK_TITLE = 20
    Y_DOMAIN = [0, (PLOT_HEIGHT - TRACK_HEADER) / PLOT_HEIGHT]

    curveXAxisPositionProps = lambda inTrackPos: dict(side="bottom", 
                                                      position=1 - (TRACK_TITLE + (inTrackPos + 1) * CURVE_HEADER)/PLOT_HEIGHT, 
                                                      anchor='free')

    curveLabelPositionProps = lambda inTrackPos: dict(xref="x domain", xanchor="center", x=0.5,
                                                      yref="y domain", yanchor="bottom", y=1,
                                                      yshift=(TRACK_HEADER - TRACK_TITLE - (inTrackPos + 1)*CURVE_HEADER + 2))


    curveNames = df.columns[1:]
    refCurveName = df.columns[0]
    fig = make_subplots(
            rows=1, 
            cols=len(curveNames) + 1, 
            shared_yaxes=True, 
            horizontal_spacing=0 
    )

    fig.append_trace(go.Scattergl(x=[0,0], y = df[refCurveName], name="depth"), 1, 1)
    __track_header(fig, TRACK_HEADER, colIdx=0)
    __track_body(fig, TRACK_HEADER, PLOT_HEIGHT - TRACK_HEADER - PLOT_HEADER, colIdx=0)
    for idx, c in enumerate(curveNames):
        __track_header(fig, TRACK_HEADER, colIdx=idx + 1)
        fig.update_xaxes(
            title = None,
            showline=True,
            nticks=4, tickfont_color=getColor(c), linecolor=getColor(c), 
            mirror=False,
            showticklabels=True,
            gridcolor="#eee",
            linewidth=1,
            **curveXAxisPositionProps(0), 
            row=1, col=idx + 2
        )
        fig.add_annotation(
            text=f"{c}({curves[c].unit})",
            font=dict(color="white", size=10),
            bgcolor=getColor(c),
            showarrow=False,
            align="center",
            visible=True,
            **curveLabelPositionProps(0),
            row=1, col=idx + 2,
        )
        trace = go.Scatter(
            x=df[c], y=df[refCurveName], name=c, line=getLineConfig(c)
        )
        fig.append_trace(trace, 1, idx + 2)
        __track_body(fig, TRACK_HEADER, PLOT_HEIGHT - TRACK_HEADER - PLOT_HEADER, colIdx=idx + 1)

    fig.update_yaxes(
        showline=False,
        linewidth=0.5,
        linecolor="#444",
        mirror=True,
        autorange="reversed",
        gridcolor="#eee",
        domain=Y_DOMAIN
    )
    fig.update_yaxes(position=0.65/(1 + len(curveNames)), showline=False, anchor='free', row=1, col=1)

    plot_title=title or f"Logplot for {','.join(curveNames)}"
    fig.update_layout(
        title=dict(text=plot_title, xanchor='center', yanchor='bottom', x=0.5, y=(PLOT_HEIGHT - PLOT_HEADER + 20)/PLOT_HEIGHT),
        plot_bgcolor="#fff",
        overwrite=True,
        showlegend=False,
        margin=dict(l=0, r=0, t=PLOT_HEADER, b=0),
        width=columnWidth * (len(curveNames) + 1),
        height=PLOT_HEIGHT,
        autosize=False,
    )
    return fig

def __makeHoverTemplate(columns):
    template = ""
    for idx,c in enumerate(columns):
        template += (f"<b>{c}</b>:" + "<span>%{customdata[" + str(idx) + "]}</span><br>")
    return template

def advLogplot(df, curves, track_styles, title = None, zoneDF = None):
    curveNames = list(df.columns[1:])
    refCurveName = df.columns[0]

    PLOT_HEADER = 60
    TRACK_HEADER = 180
    PLOT_HEIGHT = 1000
    columnWidth = 150
    CURVE_HEADER = 47
    TRACK_TITLE = 20
    Y_DOMAIN = [0, (PLOT_HEIGHT - TRACK_HEADER) / PLOT_HEIGHT]
    STATIC_TRACKS = lambda : 1 + (0 if zoneDF is None or zoneDF.empty else 1)
    TRACK_NUM = lambda: len(track_styles) + STATIC_TRACKS()
    X_DOMAIN_SIZE = lambda: 1/TRACK_NUM()
    X_DOMAIN = lambda trackIdx: [trackIdx*X_DOMAIN_SIZE(), (trackIdx + 1) * X_DOMAIN_SIZE()]

    YAXIS_PROPS = dict(showline=False,
                       linewidth=0.5,
                       linecolor="#444",
                       mirror=True,
                       #autorange="reversed",
                       gridcolor="#eee",
                       domain=Y_DOMAIN)

    YAXIS_LIMIT_PROPS = lambda df: dict(range=[
            float(df[refCurveName].iloc[0] + (df[refCurveName].iloc[-1] - df[refCurveName].iloc[0])/4),
            float(df[refCurveName].iloc[0])
        ])
    XAXIS_DEFAULT_PROPS = dict(title = None, showline=True, 
                               mirror=True, 
                               showticklabels=True, gridcolor="#eee", linewidth=1)

    curveXAxisPositionProps = lambda inTrackPos: dict(side="bottom", 
                                                      #position=1 - (TRACK_TITLE + (inTrackPos + 1) * CURVE_HEADER)/(PLOT_HEIGHT - PLOT_HEADER), 
                                                      position=1 - (TRACK_TITLE + 15 + inTrackPos * CURVE_HEADER)/(PLOT_HEIGHT - PLOT_HEADER), 
                                                      anchor='free')

    curveLabelPositionProps = lambda inTrackPos, overlaying_idx: dict(xref=f"x{overlaying_idx} domain", xanchor="center", x=0.5,
                                                      yref="y domain", yanchor="top", y=1,
                                                      yshift=TRACK_HEADER - TRACK_TITLE - inTrackPos*CURVE_HEADER - 6)

    def curveXAxisLimitProps(curveSpec):
        logType = curveSpec.get('xaxis', {}).get('scale', 'linear')
        limits = curveSpec.get('xaxis', {}).get('range', None)
        if limits:
            limits = np.array(limits)
            limits = np.log10(limits) if logType == 'log' else limits
            return dict(type=logType, range=list(limits))
        return dict(type=logType)

    def curveProps(curveSpec):
        curveProperties = { **curveSpec }
        if 'xaxis' in curveProperties:
            del curveProperties['xaxis']
        #curveProperties['name'] = ''
        return curveProperties

    fig = make_subplots( rows=1, 
            cols=TRACK_NUM(), 
            shared_yaxes=True, 
            horizontal_spacing=0 
    )
    # Prepare axes
    yaxes = { 'yaxis': { **YAXIS_PROPS, **YAXIS_LIMIT_PROPS(df) } }
    xaxes = { 'xaxis': dict(domain=X_DOMAIN(0)) }
    if zoneDF is not None and not zoneDF.empty:
        xaxes['xaxis2'] = dict( domain=X_DOMAIN(1), range=[0, 1])

    for idx in range(STATIC_TRACKS()):
        trace = go.Scattergl(x=[0,0], y = df[refCurveName].head(), name="Depth", xaxis=f"x{'' if idx==0 else (idx + 1)}", visible=False)
        fig.add_trace(trace)
        __track_header(fig, TRACK_HEADER, xdomain='' if idx == 0 else (idx + 1) , colIdx = idx, 
                       curve='DEPTH' if idx == 0 else "Zone",
                       unit='m')
        if idx == 1:
            # __drawZoneTrack 
            for _,row in zoneDF.iterrows():
                trace = go.Scattergl(x=[1,1],y=[row['start'],row['stop']], name="Zone", xaxis="x2", fill='tozerox', mode='lines', line_width=0)
                fig.add_trace(trace)
    xaxis_index = len(xaxes.keys())
    track_idx = xaxis_index

    selectedCurves = [refCurveName]
    for track_style in track_styles:
        trackConfig = getTrackConfig(track_style)
        for c in trackConfig['curves']:
            if c['name'] in curveNames:
                selectedCurves.append(c['name'])

    hoverdata = df[selectedCurves]
    hovertemplate = __makeHoverTemplate(list(hoverdata.columns))

    print(selectedCurves, curveNames)
    print(hovertemplate)
    for _,track_style in enumerate(track_styles):
        trackConfig = getTrackConfig(track_style)
        overlaying_idx = 1
        have_track = False
        for jdx, curveSpec in enumerate(trackConfig['curves']):
            c = curveSpec['name']
            if c not in curveNames:
                print(f"Curve {c} is absent in log file")
                continue
            have_track = True
            xaxis_index += 1
            xaxes[f'xaxis{xaxis_index}'] = dict(domain=X_DOMAIN(track_idx), 
                                                tickfont_color=curveSpec.get('line_color', curveSpec.get('marker_color')), 
                                                linecolor=curveSpec.get('line_color', curveSpec.get('marker_color')),
                                                **curveXAxisLimitProps(curveSpec),
                                                **curveXAxisPositionProps(jdx),
                                                **XAXIS_DEFAULT_PROPS )
            if jdx == 0:
                overlaying_idx = xaxis_index
            else:
                xaxes[f'xaxis{xaxis_index}']['overlaying'] = f'x{overlaying_idx}'

            trace = go.Scattergl(
                x=df[c], y=df[refCurveName], xaxis=f'x{xaxis_index}', **curveProps(curveSpec), 
                customdata=hoverdata, hovertemplate=hovertemplate
            )
            fig.add_trace(trace)
            fig.add_annotation(
                text=f"{c}({curves[c].unit})",
                font=dict(color="white", size=10),
                bgcolor=curveSpec.get('line_color', curveSpec.get('marker_color')),
                showarrow=False,
                align="center",
                visible=True,
                **curveLabelPositionProps(jdx, overlaying_idx),
            )
        if have_track:
            __track_header(fig, TRACK_HEADER, xdomain=overlaying_idx, colIdx=track_idx)
            __track_body(fig, TRACK_HEADER, PLOT_HEIGHT - TRACK_HEADER - PLOT_HEADER, xdomain=overlaying_idx)
            track_idx += 1

    plot_title=title or f"Logplot for {','.join(curveNames)}"
    print(xaxes)
    fig.update_layout(
        **yaxes,
        **xaxes,
        title=dict(text=plot_title, xanchor='center', yanchor='bottom', x=0.5, y=(PLOT_HEIGHT - PLOT_HEADER + 20)/PLOT_HEIGHT),
        plot_bgcolor="#fff",
        overwrite=True,
        showlegend=False,
        margin=dict(l=0, r=0, t=PLOT_HEADER, b=0),
        width=columnWidth * TRACK_NUM(),
        height=PLOT_HEIGHT,
        autosize=False,
        hoversubplots='overlaying',
        hovermode="y unified",
        hoverlabel_bgcolor="white",
        hoverlabel_grouptitlefont_lineposition='through'
    )
    #fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(**YAXIS_PROPS)
    fig.update_yaxes(position=0.60/TRACK_NUM(), side='left', showline=False, anchor='free', row=1, col=1)
    for idx in range(STATIC_TRACKS()):
        __track_body(fig, TRACK_HEADER, PLOT_HEIGHT - TRACK_HEADER - PLOT_HEADER, xdomain='' if idx == 0 else (idx + 1), colIdx=idx)
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

