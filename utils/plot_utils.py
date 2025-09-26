import os
import json
import math
import re
import hashlib
import numpy as np
import lasio
import yaml
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import write_json as __write_json
from base_utils import recursive_get, update_dict, standard_curve_name, find_similar_curves, find_star_curves
_allTrackConfigs = None
_allCurveSpecs = None
_allLogplotConfigs = None

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

def gridcolor():
    return "#aaf"

def getTrackConfig(trackstyle):
    global _allTrackConfigs
    if _allTrackConfigs is None:
        with open('utils/track_template.yaml') as file:
            _allTrackConfigs = yaml.safe_load(file)
    return _allTrackConfigs.get(trackstyle)

def getLogplotConfig(curve):
    global _allLogplotConfigs
    if _allLogplotConfigs is None:
        with open('utils/tracks4curve.yaml') as f:
            _allLogplotConfigs = yaml.safe_load(f)
    return _allLogplotConfigs.get(curve)

def getCurveSpec(curve):
    global _allCurveSpecs
    if _allCurveSpecs is None:
        with open('utils/curve.specs.yaml') as f:
            _allCurveSpecs = yaml.safe_load(f)
    return _allCurveSpecs.get(curve)

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
    print("LOGPLOTTTTTT", list(df.columns))
    
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


    curveNames = df.columns
    refCurveName = df.index.name
    fig = make_subplots(
            rows=1, 
            cols=len(curveNames) + 1, 
            shared_yaxes=True, 
            horizontal_spacing=0 
    )

    fig.append_trace(go.Scattergl(x=[0,0], y = df.index, name="depth"), 1, 1)
    __track_header(fig, TRACK_HEADER, colIdx=0, curve=df.index.name, unit=curves[df.index.name].unit)
    __track_body(fig, TRACK_HEADER, PLOT_HEIGHT - TRACK_HEADER - PLOT_HEADER, colIdx=0)
    for idx, c in enumerate(curveNames):
        curveSpec = getCurveSpec(standard_curve_name(c))
        curveProperties = {}
        curveLimitProperties = {}
        if curveSpec:
            curveProperties = {**curveSpec}
            if 'xaxis' in curveProperties:
                del curveProperties['xaxis']
                
                logType = curveSpec.get('xaxis', {}).get('scale', 'linear')
                limits = curveSpec.get('xaxis', {}).get('range', None)
                curveLimitProperties = dict(type=logType)
                if limits:
                    limits = np.array(limits)
                    limits = np.log10(limits) if logType == 'log' else limits
                    curveLimitProperties = dict(type=logType, range=list(limits))
        else:
            curveSpec = {"line_color": getHashColor(c)}
        __track_header(fig, TRACK_HEADER, colIdx=idx + 1)
        #print(curveSpec, c)
        fig.update_xaxes(
            title = None,
            showline=True,
            nticks=4, tickfont_color=curveSpec.get('line_color') or curveSpec.get('marker_color'), 
            linecolor=curveSpec.get('line_color') or curveSpec.get('marker_color'), 
            mirror=False,
            showticklabels=True,
            gridcolor=gridcolor(),
            linewidth=1,
            **curveXAxisPositionProps(0), 
            **curveLimitProperties,
            row=1, col=idx + 2
        )
        fig.add_annotation(
            text=f"{c}({curves[c].unit})",
            font=dict(color="white", size=10),
            bgcolor=curveSpec.get('line_color') or curveSpect.get('marker_color'),
            showarrow=False,
            align="center",
            visible=True,
            **curveLabelPositionProps(0),
            row=1, col=idx + 2,
        )
        trace = go.Scattergl(
            x=df[c], y=df.index, name=c, line=getLineConfig(c), **curveProperties
        )
        fig.append_trace(trace, 1, idx + 2)
        __track_body(fig, TRACK_HEADER, PLOT_HEIGHT - TRACK_HEADER - PLOT_HEADER, colIdx=idx + 1)

    fig.update_yaxes(
        showline=False,
        linewidth=0.5,
        linecolor="#444",
        mirror=True,
        autorange="reversed",
        gridcolor=gridcolor(),
        domain=Y_DOMAIN
    )
    fig.update_yaxes(position=0.65/(1 + len(curveNames)), showline=False, showgrid=False, anchor='free', row=1, col=1)

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

def __calc_new_col(df, op, operands):
    def func(row):
        ret = None
        if op == 'mul':
            for oprand in operands:
                if ret is None:
                    ret = row[oprand]
                else:
                    ret = ret * row[oprand]
        return ret
    return df.apply(func, axis = 1)

def advLogplot(df, curves, track_styles, title = None, keyZoneDF = None, zoneDF = None, select_first=False):
    curveNames = lambda : df.columns[1:]
    refCurveName = df.columns[0]
    print("--- advLogplot ----", curveNames())

    have_dup_track = (len(find_star_curves(curveNames())) > 0)

    PLOT_HEADER = 60
    #TRACK_HEADER = 180
    TRACK_HEADER = 150
    PLOT_HEIGHT = 1000
    columnWidth = 150
    #CURVE_HEADER = 47
    CURVE_HEADER = 40
    TRACK_TITLE = 20
    Y_DOMAIN = [0, (PLOT_HEIGHT - TRACK_HEADER) / PLOT_HEIGHT]
    #STATIC_TRACKS = lambda : 1 + (0 if zoneDF is None or zoneDF.empty else 1)
    STATIC_TRACKS = lambda : 2
    TRACK_NUM = lambda: len(track_styles) + STATIC_TRACKS() + (1 if have_dup_track else 0)
    X_DOMAIN_SIZE = lambda: 1/TRACK_NUM()
    X_DOMAIN = lambda trackIdx: [trackIdx*X_DOMAIN_SIZE(), (trackIdx + 1) * X_DOMAIN_SIZE()]

    YAXIS_PROPS = dict(showline=False,
                       linewidth=0.5,
                       linecolor="#444",
                       #mirror=True,
                       showgrid=True,
                       showticklabels=True,
                       gridcolor=gridcolor(),
                       #gridwidth=0.5,
                       domain=Y_DOMAIN)

    YAXIS_LIMIT_PROPS = lambda df: dict(range=[
            #float(df[refCurveName].iloc[0] + (df[refCurveName].iloc[-1] - df[refCurveName].iloc[0])/4),
            float(df[refCurveName].iloc[-1]),
            float(df[refCurveName].iloc[0])
        ])

    XAXIS_DEFAULT_PROPS = dict(title = None, showline=True, 
                               mirror=True, fixedrange=True,
                               showticklabels=False, showgrid=False, gridcolor=gridcolor(), linewidth=1)

    curveXAxisPositionProps = lambda inTrackPos: dict(side="bottom", 
                                                      position=1-(TRACK_TITLE + 15 + inTrackPos * CURVE_HEADER)/(PLOT_HEIGHT - PLOT_HEADER), 
                                                      anchor='free')

    curveLabelPositionProps = lambda inTrackPos, overlaying_idx: dict(xref=f"x{overlaying_idx} domain", xanchor="center", x=0.5,
                                                      yref="y domain", yanchor="top", y=1,
                                                      yshift=TRACK_HEADER - TRACK_TITLE - inTrackPos*CURVE_HEADER - 6)

    curveUnitPositionProps = lambda inTrackPos, overlaying_idx, side: dict(xref=f"x{overlaying_idx} domain",
                                                                           align=side, xanchor=side, 
                                                                           x=0 if side == 'left' else 1, 
                                                                           yref="y domain", yanchor="top", y=1, 
                                                                           yshift=TRACK_HEADER - TRACK_TITLE - inTrackPos*CURVE_HEADER - 28)

    def curveXAxisLimitProps(curveSpec):
        logType = curveSpec.get('xaxis', {}).get('scale', 'linear')
        limits = curveSpec.get('xaxis', {}).get('range', None)
        if limits:
            limits = np.array(limits)
            limits = np.log10(limits) if logType == 'log' else limits
            return dict(type=logType, range=list(limits))
        return dict(type=logType)

    def curveProps(curveSpec):
        removed_props = ('xaxis', 'expr', 'unit', 'fill2', 'fill2color', 'fill2name')
        curveProperties = { **curveSpec }
        for p in removed_props:
            if p in curveProperties:
                del curveProperties[p]
        #curveProperties['name'] = ''
        return curveProperties

    fig = make_subplots( rows=1, 
            cols=TRACK_NUM(), 
            #shared_yaxes=True,
            #horizontal_spacing=0.005
    )
    # Prepare axes
    yaxes = { 'yaxis': { **YAXIS_PROPS, **YAXIS_LIMIT_PROPS(df) }, 'yaxis2': { **YAXIS_PROPS, **YAXIS_LIMIT_PROPS(df) }}
    xaxes = { 'xaxis': dict(domain=X_DOMAIN(0), range=[0,1]) }
    #if zoneDF is not None and not zoneDF.empty:
    xaxes['xaxis2'] = dict( domain=X_DOMAIN(1), range=[0, 1])
    if "TVDSS" in list(df.columns):
        xaxes['xaxis3'] = {"domain":X_DOMAIN(1), **XAXIS_DEFAULT_PROPS, "overlaying":'x2'}

    for idx in range(STATIC_TRACKS()):
        trace = go.Scattergl(x=[0,0], y = df[refCurveName].head(), name="Depth", xaxis=f"x{'' if idx==0 else (idx + 1)}", visible=False)
        fig.add_trace(trace)
        __track_header(fig, TRACK_HEADER, xdomain='' if idx == 0 else (idx + 1) , colIdx = idx, 
                       curve='DEPTH' if idx == 0 else ( "TVDSS" if "TVDSS" in list(df.columns) else "ZONE" ),
                       unit='m')
        if idx == 1:
            if isinstance(keyZoneDF, pd.DataFrame):
                # __drawZoneTrack 
                for _,row in zoneDF.iterrows():
                    trace = go.Scattergl(x=[1,1],y=[row['start'],row['stop']],name="Zone", xaxis="x2", yaxis="y2", line_width=0, fill='tozerox', 
                                         mode='lines')
                    fig.add_trace(trace)
                    fig.add_annotation(xref="x2 domain", xanchor="right", x=1, xshift=-10, font_size=8, 
                                       yref="y", yanchor="middle", y=(row['start'] + row['stop'])/2, 
                                       text=row['Surface'], showarrow=False, bgcolor='#fff')
            if "TVDSS" in list(df.columns):
                tvdss = go.Scattergl(x=df['TVDSS'], y=df[refCurveName], name="TVDSS", xaxis="x3", yaxis="y2", line_width=0, mode="lines")
                fig.add_trace(tvdss)
        elif idx == 0:
            if isinstance(keyZoneDF, pd.DataFrame):
                for _,row in keyZoneDF.iterrows():
                    trace = go.Scattergl(x=[1,1],y=[row['start'],row['stop']], name="Zone", xaxis="x", line_width=0, fill='tozerox',  
                                         mode='lines')
                    fig.add_trace(trace)
                    fig.add_annotation(xref="x domain", xanchor="right", x=1, xshift=-10, font_size=8, 
                                       yref="y", yanchor="middle", y=(row['start'] + row['stop'])/2, 
                                       text=row['Surface'], showarrow=False, bgcolor='#fff')
                    fig.add_annotation(xref="x domain", xanchor="right", x=1, xshift=-10, font_size=8, 
                                       yref="y", yanchor="top", y=row['start'], yshift=-10,
                                       text=row['Surface'], showarrow=False, bgcolor='#fff')
                    fig.add_annotation(xref="x domain", xanchor="right", x=1, xshift=-10, font_size=8, 
                                       yref="y", yanchor="bottom", y=row['stop'], yshift=10,
                                       text=row['Surface'], showarrow=False, bgcolor='#fff')

    xaxis_index = len(xaxes.keys())
    track_idx = 2

    selectedCurves = [refCurveName]
    if "TVDSS" in list(df.columns):
        selectedCurves.append("TVDSS")
    for track_style in track_styles:
        trackConfig = getTrackConfig(track_style)
        print("Track config:", track_style, trackConfig)
        for c in trackConfig['curves']:
            if 'expr' in c and c['name'] not in curveNames():
                expr = c['expr']
                operands = []
                for operand in expr['operands']:
                    new_operand = None
                    for cname in curveNames():
                        #if cname == operand or re.match(rf'{operand}:.+$', cname):
                        if standard_curve_name(cname) == operand:
                            new_operand = cname
                    if new_operand is None:
                        raise Exception(f"No {operand} curve found")
                    operands.append(new_operand)
                print(operands)
                df[c['name']] = __calc_new_col(df, expr['op'], operands)
                curves[c['name']] = { "unit": c['unit']}

            sim_curves = find_similar_curves(c['name'], curveNames())
            if len(sim_curves) > 0:
                if select_first:
                    selectedCurves.append(sim_curves[0])
                else:
                    selectedCurves.append(sim_curves[-1])
            star_curves = find_star_curves(sim_curves)
            if len(star_curves) > 0:
                selectedCurves.append(star_curves[0])

    hoverdata = df[list(set(selectedCurves))]
    hovertemplate = __makeHoverTemplate(list(hoverdata.columns))

    for track_style in track_styles:
        trackConfig = getTrackConfig(track_style)
        overlaying_idx = None
        have_track = False
        _jdx = 0
        dup_track_curves = []
        for curveSpec in trackConfig['curves']:
            c = curveSpec['name']
            sim_curves = find_similar_curves(c, curveNames())
            if len(sim_curves) == 0:
                print(f"Curve {c} is absent in log file")
                continue
            if select_first:
                c = sim_curves[0]
            else:
                c = sim_curves[-1]
            star_curves = find_star_curves(sim_curves)
            if len(star_curves) > 0:
                dup_track_curves.append({'name': star_curves[0], 'spec': curveSpec})

            have_track = True
            xaxis_index += 1
            xaxes[f'xaxis{xaxis_index}'] = dict(domain=X_DOMAIN(track_idx), 
                                                tickfont_color=curveSpec.get('line_color', curveSpec.get('marker_color')), 
                                                linecolor=curveSpec.get('line_color', curveSpec.get('marker_color')),
                                                **curveXAxisLimitProps(curveSpec),
                                                **curveXAxisPositionProps(_jdx),
                                                **XAXIS_DEFAULT_PROPS )
            if _jdx == 0:
                overlaying_idx = xaxis_index
            else:
                xaxes[f'xaxis{xaxis_index}']['overlaying'] = f'x{overlaying_idx}'

            if curveSpec.get('fill2', None) == 'tonextx':
                axisProps = curveXAxisLimitProps(curveSpec)
                v = np.max(axisProps.get('range', [0, 1000]))
                maxtrace = go.Scattergl(
                    x=[v]*len(df.index), y=df[refCurveName], xaxis=f'x{xaxis_index}', yaxis=f'y{track_idx + 1}', mode='lines', 
                    fill='tozerox', fillcolor=curveSpec.get('fill2color')
                )
                fig.add_trace(maxtrace)

            cprops = curveProps(curveSpec)
            trace = go.Scattergl(
                x=df[c], y=df[refCurveName], xaxis=f'x{xaxis_index}', yaxis=f'y{track_idx + 1}', **cprops, 
                customdata=hoverdata, hovertemplate=hovertemplate
            )
            fig.add_trace(trace)

            # Curve label
            fig.add_annotation(
                text=f"{c}({curves[c]['unit']})",
                font=dict(color="white", size=10),
                bgcolor=curveSpec.get('line_color', curveSpec.get('marker_color')),
                showarrow=False,
                align="center",
                visible=True,
                **curveLabelPositionProps(_jdx, overlaying_idx)
            )
            # Curve limits
            fig.add_annotation(text=f"{recursive_get(curveSpec, ['xaxis', 'range'])[0]:.1f}", 
                               font_color=curveSpec.get('line_color', curveSpec.get('marker_color')),
                               font_size = 10, showarrow=False, visible=True, 
                               **curveUnitPositionProps(_jdx, overlaying_idx, 'left'))
            fig.add_annotation(text=f"{recursive_get(curveSpec, ['xaxis', 'range'])[1]:.1f}", 
                               font_color=curveSpec.get('line_color', curveSpec.get('marker_color')),
                               font_size = 10, showarrow=False, visible=True, 
                               **curveUnitPositionProps(_jdx, overlaying_idx, 'right'))
            if curveSpec.get('fill2', None) == 'tonextx':
                _jdx += 1
                fig.add_shape(
                            type='rect',
                            xref=f"x{overlaying_idx} domain", x0=0.01, x1=0.99, 
                            yref=f"y domain", ysizemode='pixel',yanchor=1,
                            y0=TRACK_HEADER - 2*TRACK_TITLE - _jdx*CURVE_HEADER - 6,
                            y1=TRACK_HEADER - TRACK_TITLE - _jdx*CURVE_HEADER - 6,
                            visible=True, line_color='#c0c0c0', line_width=1, fillcolor=curveSpec.get('fill2color'),
                            layer='above'
                            )
                fig.add_annotation(text=curveSpec.get('fill2name'), 
                               font=dict(color=curveSpec.get('line_color', curveSpec.get('marker_color')), size=10),
                               bgcolor=curveSpec.get('fill2color'),
                               showarrow=False,
                               align="center",
                               visible=True,
                               **curveLabelPositionProps(_jdx, overlaying_idx))
            # Track grid
            limits = recursive_get(curveSpec, ['xaxis' , 'range']) or [0.0, 1.0]
            nticks = recursive_get(curveSpec, ['xaxis' , 'nticks']) or 5
            step = abs(limits[1] - limits[0])/nticks
            ticks = recursive_get(curveSpec, ['xaxis', 'ticks']) or [(i + 1)*step for i in range(nticks - 1)]
            if _jdx == 0:
                for x in ticks:
                    fig.add_shape(type='line', 
                                  xref=f"x{overlaying_idx}", x0=x, x1=x,
                                  yref='y domain', y0=0, y1=1,
                                  line_width=0.5, line_color=gridcolor(), layer="below")
            _jdx += 1
            
        if have_track:
            __track_header(fig, TRACK_HEADER, xdomain=overlaying_idx, colIdx=track_idx)
            __track_body(fig, TRACK_HEADER, PLOT_HEIGHT - TRACK_HEADER - PLOT_HEADER, xdomain=overlaying_idx)
            yaxes[f'yaxis{track_idx + 1}'] = {**yaxes['yaxis'], "matches": 'y', "anchor":f'x{overlaying_idx}', "showticklabels":False}
            track_idx += 1

        # Draw duptrack
        if len(dup_track_curves) > 0:
            print('HAVE DUP TRACK, ......', dup_track_curves)
            _jdx = 0
            for c in dup_track_curves:
                curveSpec = c['spec']
                cName = c['name']
                xaxis_index += 1
                xaxes[f'xaxis{xaxis_index}'] = dict(domain=X_DOMAIN(track_idx), 
                                                    tickfont_color=curveSpec.get('line_color', curveSpec.get('marker_color')), 
                                                    linecolor=curveSpec.get('line_color', curveSpec.get('marker_color')),
                                                    **curveXAxisLimitProps(curveSpec),
                                                    **curveXAxisPositionProps(_jdx),
                                                    **XAXIS_DEFAULT_PROPS )
                if _jdx == 0:
                    overlaying_idx = xaxis_index
                else:
                    xaxes[f'xaxis{xaxis_index}']['overlaying'] = f'x{overlaying_idx}'

                if curveSpec.get('fill2', None) == 'tonextx':
                    axisProps = curveXAxisLimitProps(curveSpec)
                    v = np.max(axisProps.get('range', [0, 1000]))
                    maxtrace = go.Scattergl(
                        x=[v]*len(df.index), y=df[refCurveName], xaxis=f'x{xaxis_index}', yaxis=f'y{track_idx + 1}', mode='lines', 
                        fill='tozerox', fillcolor=curveSpec.get('fill2color')
                    )
                    fig.add_trace(maxtrace)

                cprops = curveProps(curveSpec)
                trace = go.Scattergl(
                    x=df[cName], y=df[refCurveName], xaxis=f'x{xaxis_index}', yaxis=f'y{track_idx + 1}', **cprops, 
                    customdata=hoverdata, hovertemplate=hovertemplate
                )
                fig.add_trace(trace)
                # Curve label
                fig.add_annotation(
                    text=f"{cName}({curves[cName]['unit']})",
                    font=dict(color="white", size=10),
                    bgcolor=curveSpec.get('line_color', curveSpec.get('marker_color')),
                    showarrow=False,
                    align="center",
                    visible=True,
                    **curveLabelPositionProps(_jdx, overlaying_idx)
                )
                # Curve limits
                fig.add_annotation(text=f"{recursive_get(curveSpec, ['xaxis', 'range'])[0]:.1f}", 
                                   font_color=curveSpec.get('line_color', curveSpec.get('marker_color')),
                                   font_size = 10, showarrow=False, visible=True, 
                                   **curveUnitPositionProps(_jdx, overlaying_idx, 'left'))
                fig.add_annotation(text=f"{recursive_get(curveSpec, ['xaxis', 'range'])[1]:.1f}", 
                                   font_color=curveSpec.get('line_color', curveSpec.get('marker_color')),
                                   font_size = 10, showarrow=False, visible=True, 
                                   **curveUnitPositionProps(_jdx, overlaying_idx, 'right'))
                if curveSpec.get('fill2', None) == 'tonextx':
                    _jdx += 1
                    fig.add_shape(
                                type='rect',
                                xref=f"x{overlaying_idx} domain", x0=0.01, x1=0.99, 
                                yref=f"y domain", ysizemode='pixel',yanchor=1,
                                y0=TRACK_HEADER - 2*TRACK_TITLE - _jdx*CURVE_HEADER - 6,
                                y1=TRACK_HEADER - TRACK_TITLE - _jdx*CURVE_HEADER - 6,
                                visible=True, line_color='#c0c0c0', line_width=1, fillcolor=curveSpec.get('fill2color'),
                                layer='above'
                                )
                    fig.add_annotation(text=curveSpec.get('fill2name'), 
                                   font=dict(color=curveSpec.get('line_color', curveSpec.get('marker_color')), size=10),
                                   bgcolor=curveSpec.get('fill2color'),
                                   showarrow=False,
                                   align="center",
                                   visible=True,
                                   **curveLabelPositionProps(_jdx, overlaying_idx))
                # Track grid
                limits = recursive_get(curveSpec, ['xaxis' , 'range']) or [0.0, 1.0]
                nticks = recursive_get(curveSpec, ['xaxis' , 'nticks']) or 5
                step = abs(limits[1] - limits[0])/nticks
                ticks = recursive_get(curveSpec, ['xaxis', 'ticks']) or [(i + 1)*step for i in range(nticks - 1)]
                if _jdx == 0:
                    for x in ticks:
                        fig.add_shape(type='line', 
                                      xref=f"x{overlaying_idx}", x0=x, x1=x,
                                      yref='y domain', y0=0, y1=1,
                                      line_width=0.5, line_color=gridcolor(), layer="below")
                _jdx += 1
            
            __track_header(fig, TRACK_HEADER, xdomain=overlaying_idx, colIdx=track_idx)
            __track_body(fig, TRACK_HEADER, PLOT_HEIGHT - TRACK_HEADER - PLOT_HEADER, xdomain=overlaying_idx)
            yaxes[f'yaxis{track_idx + 1}'] = {**yaxes['yaxis'], "matches": 'y', "anchor":f'x{overlaying_idx}', "showticklabels":False}
            track_idx += 1
            dup_track_curves = []

    plot_title=title or f"Logplot for {','.join(curveNames())}"
    update_dict(yaxes['yaxis'], dict(position=0.60/TRACK_NUM(), side='left', showgrid=False, showline=False, anchor='free'))
    update_dict(yaxes['yaxis2'], dict(position=1.60/TRACK_NUM(), matches='y', side='left', showgrid=False, showline=False, anchor='free'))
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
    for idx in range(STATIC_TRACKS()):
        __track_body(fig, TRACK_HEADER, PLOT_HEIGHT - TRACK_HEADER - PLOT_HEADER, xdomain='' if idx == 0 else (idx + 1), colIdx=idx)

    return fig

def write_json(fig, path):
    with open(path, 'w') as f:
        __write_json(fig, f)
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

def multi_chart(chart_titles, traces, chart_titles1 = [], traces1 = [], main_title=""):
    colors = ['blue', 'red', 'green', 'purple', 'magenta']
    suffixes = ['Production', 'Train', 'Predict', 'Predict-Future', 'Injection']
    colors1 = ['magenta']
    suffixes1 = ['Injection']
    chart_cnt = len(chart_titles) + len(chart_titles1)
    #print(chart_cnt)
    fig = make_subplots(
        cols=1, rows=chart_cnt, shared_xaxes=True,
        subplot_titles=chart_titles + chart_titles1
    )
    for i,title in enumerate(chart_titles):
        for j,tr in enumerate(traces):
            trace = go.Scatter(x = tr['x'], y = tr['y'][:, i], 
                            name=f"[{title}] {suffixes[j]}", mode='lines', line=dict(color=colors[j]), legendgroup=f'{i}')
            fig.append_trace(trace, i+1, 1)

    for i,title in enumerate(chart_titles1):
        for j,tr in enumerate(traces1):
            trace = go.Scatter(x = tr['x'], y = tr['y'][:, i], 
                            name=f"[{title}] {suffixes1[j]}", mode='lines', line=dict(color=colors1[j]), legendgroup=f'{i + len(chart_titles)}')
            fig.append_trace(trace, len(chart_titles) + i+1, 1)
    
    fig.update_xaxes(
        showline=True,
        linewidth=0.5,
        linecolor="#444",
        showticklabels=True,
        gridcolor=gridcolor(),
    )
    fig.update_yaxes(
        title_text='Oil rate (ton/day)',
        showline=True,
        linewidth=0.5,
        linecolor="#444",
        gridcolor=gridcolor(),
    )
    fig.update_layout(
        title=dict(text=main_title),
        plot_bgcolor="#fff",
        width=800,
        height=chart_cnt * 200 + 200,
        legend_tracegroupgap=150
    )
    return fig

def production_by_oilcum_chart(df_wells, all_cols, modes=["lines"]):
    PLOT_WIDTH = 960
    X_COL = 0 # xparam
    xparam = all_cols[0]
    COLORS = {
        "CV.WaterProdCum/1000": "magenta", # darkcyan
        "CV.WaterInjCum/1000": '#00FF00', # dark lime
        "CV.WaterInj_Rate": 'brown',
        "CV.WaterRate": '#008080', 
        "CV.LiqRate": "#008000",
        "CV.Watercut": "#0000ff",
        "CV.Oilcum/1000": "#800000",
        "CV.OilRate": "#ff0000",
    }
    UNITS = {
        "CV.WaterProdCum/1000": "thous,m3",
        "CV.WaterInjCum/1000": 'thous,m3',
        "CV.WaterRate": 'm3/d', 
        "CV.LiqRate": "ton/d",
        "CV.WaterInj_Rate": 'm3/d',
        "CV.Watercut": '%',
        "CV.Oilcum/1000": 'thous,ton',
        "CV.OilRate": "ton/d"
    }
    cols = all_cols[2:]
    fig = go.Figure()
    fig.update_layout(title=dict(
        text=f"Production crossplot for {",".join([w[0] for w in  df_wells ])}",
        x=0.5, xanchor='center'
    ))
    num_params = len(cols)
    half_num_params = round(num_params / 2)
    AXIS_WIDTH = 60
    WELLS = len(df_wells)
    #X_START_POS = 0.3
    X_START_POS = AXIS_WIDTH * half_num_params / PLOT_WIDTH
    Y_DOMAIN = lambda well_idx: [well_idx  / WELLS, (well_idx + 0.85) / WELLS ] if WELLS > 1 else [0, 1]
    yaxis_idx = 0
    overlaying_idx = 0
    for well_idx, (well, df_well) in enumerate(df_wells):
        x_suffix = str(well_idx + 1)
        if x_suffix == "1":
            x_suffix = ""
        xaxis_key = f"xaxis{x_suffix}"
        xaxis_name = f"x{x_suffix}"
        if well_idx > 0:
            fig.update_layout({
                xaxis_key: dict(domain=[X_START_POS, 1 - X_START_POS], 
                                title_text=f"Well {well} {xparam} ({UNITS.get(xparam, 'NA')})", showticklabels=True, 
                                #showgrid=True, gridcolor='#ccc', 
                                showgrid=True, 
                                #matches='x',
                                showline=False, linewidth=1, linecolor='#888')
            })
        else:
            fig.update_layout({
                xaxis_key: dict(domain=[X_START_POS, 1 - X_START_POS], 
                                title_text=f"Well {well} {xparam} ({UNITS.get(xparam, 'NA')})", showticklabels=True, 
                                #showgrid=True, gridcolor='#ccc',
                                showgrid=True,
                                showline=False, linewidth=1, linecolor='#888',
                                anchor="y")
            })
        overlaying_y = lambda overlaying_idx: f"y{overlaying_idx + 1 if overlaying_idx > 0 else ''}"
        for param_idx, param in enumerate(cols):
            if param_idx == 0:
                overlaying_idx = yaxis_idx
            color = COLORS[param] if param in COLORS else getColor(param)
            y_suffix = '' if yaxis_idx == 0 else str(yaxis_idx + 1)
            yaxis_name = f"y{y_suffix}"
            yaxis_key = f"yaxis{y_suffix}"
            
            if param_idx % 2 == 0:
                yaxis_props = dict(
                    side='left',
                    anchor="free",
                    overlaying=(None if param_idx == 0 else overlaying_y(overlaying_idx)),
                    position=(half_num_params - param_idx / 2) / half_num_params * X_START_POS
                )
                fig.update_layout(
                    {
                        yaxis_key: dict( title=None,
                            tickfont=dict(color=color),
                            zeroline=False,
                            showline=True, linecolor=color, linewidth=0.5,
                            domain=Y_DOMAIN(well_idx),
                            range=[0, None],
                            #showgrid=True, 
                            #gridcolor='#ccc', gridwidth=0.5,
                            showgrid=(param_idx == 0),
                            **yaxis_props
                        )
                    }
                )
                fig.add_annotation(text=f"{param_idx}-{param} ({UNITS.get(param, 'NA')})", textangle=-90, font=dict(color=color),
                    x=(half_num_params - param_idx / 2) / half_num_params * X_START_POS, 
                    xref='paper', xanchor='right', align = 'center', xshift=-20,
                    showarrow=False,
                    y=0.5, yref=f"{yaxis_name} domain", yanchor='middle')
            else:
                yaxis_props = dict(
                    side='right',
                    anchor="free",
                    overlaying=(None if param_idx == 0 else overlaying_y(overlaying_idx)),
                    position= 1 - X_START_POS + (param_idx - 1) / 2 / half_num_params * X_START_POS
                )
                fig.update_layout(
                    {
                        yaxis_key: dict( title=None,
                            tickfont=dict(color=color),
                            zeroline=False,
                            showline=True, linecolor=color, linewidth=0.5,
                            domain=Y_DOMAIN(well_idx),
                            range=[0, None],
                            #showgrid=True, 
                            #gridcolor='#ccc', gridwidth=0.5,
                            showgrid=(param_idx == 0),
                            **yaxis_props
                        )
                    }
                )
                fig.add_annotation(text=f"{param_idx}-{param} ({UNITS.get(param, 'NA')})", textangle=-90, font=dict(color=color),
                    x=1 - X_START_POS + (param_idx - 1) / 2 / half_num_params * X_START_POS, 
                    xref='paper', xanchor='right', align = 'center', xshift=40,
                    showarrow=False,
                    y=0.5, yref=f"{yaxis_name} domain", yanchor='middle')

            fig.add_trace(
                go.Scattergl(
                    x=df_well[all_cols[X_COL]],
                    y=df_well[param],
                    name=f"{param}",
                    mode="+".join(modes),
                    line=dict(color=color),
                    xaxis=xaxis_name,
                    yaxis=yaxis_name,
                    legendgroup=well
                )
            )
            yaxis_idx = yaxis_idx + 1
        if well_idx > 0:
            fig.update_layout({
                xaxis_key: dict(anchor=f"y{overlaying_idx + 1}")
            })

    fig.update_layout(height=500 * len(df_wells), width = PLOT_WIDTH,
                        #plot_bgcolor='#fff',
                        showlegend=False,
                        legend_tracegroupgap=260,
                        legend_traceorder="grouped+reversed")
    return fig

def production_by_time_chart(df_wells, all_cols, modes = ['lines']):
    PLOT_WIDTH = 960
    DATE_COL = 0
    '''
    COLORS = {
        "CV.OilRate": "#ff0000",
        "CV.LiqRate": "#008000",
        "CV.Watercut": "#0000ff",
        "CV.Oilcum/1000": "#800000",
    }
    UNITS = {
        "CV.OilRate": "m3/month",
        "CV.LiqRate": "m3/month",
        "CV.Watercut": "%",
        "CV.Oilcum/1000": "thous,m3",
    }
    '''
    COLORS = {
        "CV.WaterProdCum/1000": "magenta", # darkcyan
        "CV.WaterInjCum/1000": '#00FF00', # dark lime
        "CV.WaterInj_Rate": 'brown',
        "CV.WaterRate": '#008080', 
        "CV.LiqRate": "#008000",
        "CV.Watercut": "#0000ff",
        "CV.Oilcum/1000": "#800000",
        "CV.OilRate": "#ff0000",
    }
    UNITS = {
        "CV.WaterProdCum/1000": "thous,m3",
        "CV.WaterInjCum/1000": 'thous,m3',
        "CV.WaterRate": 'm3/d', 
        "CV.LiqRate": "ton/d",
        "CV.WaterInj_Rate": 'm3/d',
        "CV.Watercut": '%',
        "CV.Oilcum/1000": 'thous,ton',
        "CV.OilRate": "ton/d"
    }
    cols = all_cols[2:]
    fig = go.Figure()
    fig.update_layout(title=dict(text=f"Production time chart for {",".join([w[0] for w in  df_wells ])}", 
                                 x=0.5, xanchor='center'),
                      margin=dict(t=30,b=0))    

    num_params = len(cols)
    half_num_params = round(num_params / 2)
    AXIS_WIDTH = 60
    WELLS = len(df_wells)

    #X_START_POS = 0.3
    X_START_POS = AXIS_WIDTH * half_num_params / PLOT_WIDTH
    Y_DOMAIN = lambda well_idx: [well_idx  / WELLS, (well_idx + 0.85) / WELLS ] if WELLS > 1 else [0, 1]
    yaxis_idx = 0
    overlaying_idx = 0
    for well_idx, (well, df_well) in enumerate(df_wells):
        x_suffix = str(well_idx + 1)
        if x_suffix == "1":
            x_suffix = ""
        xaxis_key = f"xaxis{x_suffix}"
        xaxis_name = f"x{x_suffix}"
        if well_idx > 0:
            fig.update_layout({
                xaxis_key: dict(domain=[X_START_POS, 1 - X_START_POS], 
                                title_text=f"Well {well}", showticklabels=True, showgrid=True, #gridcolor='#ccc', 
                                matches='x',
                                showline=True, mirror=True,linewidth=1, linecolor='#888')
            })
        else:
            fig.update_layout({
                xaxis_key: dict(domain=[X_START_POS, 1 - X_START_POS], 
                                title_text=f"Well {well}", showticklabels=True, showgrid=True, #gridcolor='#ccc',
                                showline=False, linewidth=1, linecolor='#888',
                                anchor="y")
            })
        overlaying_y = lambda overlaying_idx: f"y{overlaying_idx + 1 if overlaying_idx > 0 else ''}"
        for param_idx, param in enumerate(cols):
            if param_idx == 0:
                overlaying_idx = yaxis_idx
            color = COLORS[param] if param in COLORS else getColor(param)
            y_suffix = '' if yaxis_idx == 0 else str(yaxis_idx + 1)
            yaxis_name = f"y{y_suffix}"
            yaxis_key = f"yaxis{y_suffix}"
            if param_idx % 2 == 0:
                fig.update_layout(
                    {
                        yaxis_key: dict( title=None,
                            tickfont=dict(color=color),
                            zeroline=False,
                            showline=True, linecolor=color, linewidth=0.5,
                            domain=Y_DOMAIN(well_idx),
                            range=[0, None],
                            #gridcolor='#ccc', gridwidth=0.5,
                            showgrid=(param_idx == 0),
                            anchor="free", side='left',
                            overlaying=(None if param_idx == 0 else overlaying_y(overlaying_idx)),
                            position=(half_num_params - param_idx / 2) / half_num_params * X_START_POS
                        )
                    }
                )
                fig.add_annotation(text=f"{param_idx}-{param} ({UNITS.get(param, 'NA')})", textangle=-90, font=dict(color=color),
                    x=(half_num_params - param_idx / 2) / half_num_params * X_START_POS, 
                    xref='paper', xanchor='right', align = 'center', xshift=-20,
                    showarrow=False,
                    y=0.5, yref=f"{yaxis_name} domain", yanchor='middle')
            else:
                fig.update_layout(
                    {
                        yaxis_key: dict( title=None,
                            tickfont=dict(color=color),
                            zeroline=False,
                            showline=True, linecolor=color, linewidth=0.5,
                            domain=Y_DOMAIN(well_idx),
                            range=[0, None],
                            #gridcolor='#ccc', gridwidth=0.5,
                            showgrid=(param_idx == 0),
                            anchor="free", side='right',
                            overlaying=(None if param_idx == 0 else overlaying_y(overlaying_idx)),
                            position=1 - X_START_POS + (param_idx - 1) / 2 / half_num_params * X_START_POS
                        )
                    }
                )
                fig.add_annotation(text=f"{param_idx}-{param} ({UNITS.get(param, 'NA')})", textangle=-90, font=dict(color=color),
                    x=1 - X_START_POS + (param_idx - 1) / 2 / half_num_params * X_START_POS, 
                    xref='paper', xanchor='right', align = 'center', xshift=40,
                    showarrow=False,
                    y=0.5, yref=f"{yaxis_name} domain", yanchor='middle')

            fig.add_trace(
                go.Scattergl(
                    x=df_well[all_cols[DATE_COL]],
                    y=df_well[param],
                    name=f"{param}",
                    mode="+".join(modes),
                    line=dict(color=color),
                    xaxis=xaxis_name,
                    yaxis=yaxis_name,
                    legendgroup=well
                )
            )
            yaxis_idx = yaxis_idx + 1
        if well_idx > 0:
            fig.update_layout({
                xaxis_key: dict(anchor=f"y{overlaying_idx + 1}")
            })

    fig.update_layout(height=500 * len(df_wells), width=PLOT_WIDTH,
                        #plot_bgcolor='#fff',
                        showlegend=False,
                        legend_tracegroupgap=260,
                        legend_traceorder="grouped+reversed")
    return fig

def plot_charts(dfs, modes = ['lines+markers']):
    fig = make_subplots(cols = 1, rows = len(dfs), subplot_titles=list(dfs.keys()))
    for idx,(key,df) in enumerate(dfs.items()):
        columns = df.columns
        xcol = columns[0]
        ycols = columns[1:]
        for ycol in ycols:
            _df = df[[xcol, ycol]].dropna(how='any')
            trace = go.Scattergl(x=_df[xcol], y=_df[ycol], name=f"pressure-{ycol}", mode="+".join(modes))
            fig.append_trace(trace, row=idx+1, col=1)
        fig.update_xaxes(title_text="Date", row=idx+1, col=1)
        fig.update_yaxes(title_text="Pressure", row=idx+1, col=1)

    fig.update_layout(title=dict(text=f"Welltest data", 
                                 x=0.5, xanchor='center'))

    return fig

def __scatter_pie(radius, x, y, sectors, colors = ["rgba(255, 0, 255, 0.7)","rgba(0,0, 128, 0.7)"], names=['',''], showlegend=False):
    angles = [360 * val for val in sectors]
    if len(angles) == 0:
        angles = [360, 0]
    if all([math.isnan(a) for a in angles]):
        print("skip", angles)
        return None
    angles = [math.radians(x) for x in angles]
    
    nop = 50 #number of points

    start_angle = 0
    
    pie_segments = []
    
    for i,j,name in zip(angles,colors, names):

        end_angle = start_angle + i

        d_theta = (end_angle - start_angle) / (nop - 1)

        theta = [start_angle + i * d_theta for i in range(nop)]
        x_segment = [x + radius * math.cos(t) for t in theta]
        y_segment = [y + radius * math.sin(t) for t in theta]

        # Add the center coordinates to close the circle segment
        x_segment.append(x)
        y_segment.append(y)

        trace = go.Scattergl(name = name, x=x_segment,
                           y=y_segment,
                           mode='lines',
                           fill="toself",
                           fillcolor=j,
                           line=dict(color='blue', width=0),
                           hoverinfo='skip',
                           showlegend=showlegend and (name is not None) and (len(name) > 0)
                           )

        pie_segments.append(trace)
        start_angle = end_angle
        
    return pie_segments

def pie_map(df, groups, names, key_col=0, x_col=1, y_col=2, anno_cols=[], anno_suffixes=[], radius_cols = None, hovertemplate=None, color_palettes=None, plot_title='Pie map'):
    xSeries = df.iloc[:, x_col]
    minV = xSeries.min()
    maxV = xSeries.max()
    count = len(df.index)
    #min_radius = (maxV - minV) / count * 0.1
    #min_radius = 0
    #max_radius = (maxV - minV) / count * 1.2
    min_radius = 80
    max_radius = 300

    data = []
    #radius = (maxV - minV) / count 
    radius = 150
    for idx,group in enumerate(groups):
        first_row = True
        for row in df.itertuples(index=False):
            if radius_cols and radius_cols[idx]:
                radius = min_radius + (max_radius - min_radius) * row[radius_cols[idx]]
            if radius_cols:
                if math.isnan(row[radius_cols[idx]]) or row[radius_cols[idx]] < 1e-5:
                    continue
            pie = None
            if color_palettes and color_palettes[idx] and len(color_palettes[idx]):
                pie = __scatter_pie(radius, row[x_col], row[y_col], 
                            [row[i] for i in group], 
                            names=names[idx], 
                            colors=color_palettes[idx],
                            showlegend=first_row)
            else:
                pie = __scatter_pie(radius, row[x_col], row[y_col], 
                            [row[i] for i in group],
                            names=names[idx],
                            showlegend=first_row)
            if pie:
                data = data + pie
                first_row = False

    fig = go.Figure( data, {} )
    fig.add_trace(
        go.Scattergl(name='Well position', x=df.iloc[:, x_col], y = df.iloc[:, y_col], 
            mode="markers", 
            #marker=dict(size=16, symbol='hexagram-dot', color='lightgreen', line=dict(color="maroon", width=2)),
            marker=dict(size=4, symbol='circle', color='maroon'),
            customdata=df, 
            hovertemplate=hovertemplate)
    )
    for row in df.itertuples(index=False):
        anno_values = [ f"{0 if math.isnan(row[anno_col]) else row[anno_col]:.0f}{anno_suffixes[idx]}" for idx,anno_col in enumerate(anno_cols) ]
        anno_div = "_" * 5 * len(anno_cols)
        anno_text = f'{row[key_col]}<br><sup>{anno_div}</sup><br>{"/".join(anno_values)}'
        
        fig.add_annotation(text=anno_text, visible=True, showarrow=False,
                font=dict(color="pink", size=12, shadow='2px 2px 1px rgba(0,0,0,0.7)'),
                x=row[x_col] , y=row[y_col], xanchor='center', yanchor='top', yshift=-10, xref='x', yref='y')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(scaleanchor='x', showticklabels=False)
    fig.update_layout(title=dict(text=plot_title, x=0.5, xanchor='center'), 
            legend=dict(itemclick=False, itemdoubleclick=False), dragmode='pan')
    return fig
