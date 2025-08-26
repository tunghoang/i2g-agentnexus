import os
import json
import time
import lasio
import traceback
import numpy as np
import pandas as pd
from glob import glob, iglob
from datetime import datetime
from typing import List, Dict, Any
from multiprocessing import Process, Event

from df_utils import fill_zones
from store import Store
from naming import Naming
from cache import MemoryCache
from calendar import monthrange
from pywaterflood import CRM
from xlsx_utils import XLSX
from config.settings import DataConfig
from utils.missing_pay_utils import get_well_checklist, get_well_checklist_curves,\
    make_pseudo_log, get_training_result, remove_training_result, get_wells_has_curve, \
    read_curves_from_las, read_curves_meta_data_from_las, \
    get_runs, get_curves_in_well
from utils.plot_utils import multi_chart, advLogplot, logplot, write_json
from xlsx_utils import XLSX
from multiprocessing import Process
from mlflow.artifacts import download_artifacts

from base_utils import iframe, excel_link, getLogRules, PUBLISH_BASE, find_similar_curves

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
            track_templates = input_data.get("track_templates", "GR,LLD,NPHI")
            track_templates = track_templates.split(",")
            track_templates = [tpl.strip() for tpl in track_templates]

            df = read_curves_from_las(well, [])
            df = df.reset_index()
            if track_templates:
                keyZoneDF, zoneDF = XLSX.extract_zones1(well)
                perforationDF = XLSX.extract_perforation(well)
                perforationDF = fill_zones(perforationDF, 'md', df[df.columns[0]])
                df['PERF'] = perforationDF['PERF']
                las_curves = dict(read_curves_meta_data_from_las(well))
                las_curves['PERF'] = dict(unit='N/A')
                fig = advLogplot(df, las_curves, track_styles=track_templates, title=f"Well {well} Logplot", keyZoneDF = keyZoneDF, zoneDF=zoneDF)
            else:
                fig = logplot(df, las_curves)

            raw_path = f"{well}_planset"
            dest_path = Naming.dest_path(raw_path, category='logplot', format='json')
            publish_path = Naming.publish_path(raw_path, category='logplot', format='json')
            write_json(fig, dest_path)
            Naming.gen_site()
            return {
                "text": iframe(f'{Naming.publish_path("view_plot")}?plot={publish_path}', height='1200px')
            }
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
            well_names_input = [w.strip() for w in well_names_input]
            (
                well_names,
                loai_gieng_result,
                ten_gian_result,
                kb_result,
                nam_khoan_result,
                mong_result,
                day_md_result,
                day_tvdss_result,
                doi_tuong_khoan_result,
                log_result,
                devi_result,
                mudlog_result,
                marker_result,
                thu_via_result,
                plt_result,
                kqdvl_result,
            ) = get_well_checklist(
                wells=well_names_input, marker_path=Naming.default_marker_file()
            )
            count = len(well_names)
            out_file_relative_path = os.path.join(
                f"well_checklist_table{"_".join(well_names[:4])}.html"
            )
            out_file_path = os.path.join("/tmp", out_file_relative_path)
            df = pd.DataFrame(
                data={
                    "STT": range(1, count + 1),
                    "Tên giếng": well_names,
                    "Thân": ["N/A"] * count,
                    "Loại giếng": loai_gieng_result,
                    "Tên giàn": ten_gian_result,
                    "KB (m)": kb_result,
                    "Năm khoan": nam_khoan_result,
                    "Móng": mong_result,
                    "Đáy giếng MD": day_md_result,
                    "Đáy giếng TVDss": day_tvdss_result,
                    "Đối tượng khoan": doi_tuong_khoan_result,
                    "Log": log_result,
                    "Deviation": devi_result,
                    "Mudlog": mudlog_result,
                    "Marker": marker_result,
                    "Thử vỉa": thu_via_result,
                    "PLT": plt_result,
                    "KQĐVL": kqdvl_result,
                }
            )

            table = df.to_html(index=False)
            template = open("templates/well_checklist_tpl.html", "r").read()
            result = template.replace("{{TABLE}}", table)
            with open(out_file_path, "w") as f:
                f.write(result)

            return {"text": iframe(out_file_relative_path)}
        except Exception as e:
            traceback.print_exc()
            return {"text": f"Tool failed: {str(e)}"}

    @mcp_server.tool(
        name="well_checklist_curves",
        description="get checklist curves of well logs data",
    )
    def well_checklist_curves(**kwargs):
        try:
            input_data = json.loads(kwargs["input"])
            well_names_input: list[str] = input_data.get("wells")
            (
                well_names,
                gr_result,
                sp_result,
                cal_result,
                deep_res_result,
                med_res_result,
                shal_res_result,
                micro_res_result,
                density_result,
                neutron_result,
                sonic_result,
                pe_result,
            ) = get_well_checklist_curves(wells=well_names_input)
            out_file_relative_path = os.path.join(
                f"well_checklist_curves{"_".join(well_names[:5])}.html"
            )
            out_file_path = os.path.join("/tmp", out_file_relative_path)
            df = pd.DataFrame(
                data={
                    "Tên giếng": well_names,
                    "GammaRay": gr_result,
                    "SP": sp_result,
                    "Caliper": cal_result,
                    "DeepRes": deep_res_result,
                    "MedRes": med_res_result,
                    "ShalRes": shal_res_result,
                    "MicroRes": micro_res_result,
                    "Density": density_result,
                    "Neutron": neutron_result,
                    "Sonic": sonic_result,
                    "PE": pe_result,
                }
            )

            table = df.to_html(index=False)
            template = open("templates/well_checklist_curves_tpl.html", "r").read()
            result = template.replace("{{TABLE}}", table)
            with open(out_file_path, "w") as f:
                f.write(result)

            return {"text": out_file_relative_path}
        except Exception as e:
            traceback.print_exc()
            return {"text": f"Tool failed: {str(e)}"}

    @mcp_server.tool(
        name="create_wells_tvdss",
        description="Create file TVDSS.csv for a list of wells with TVD and TVDss",
    )
    def create_wells_tvdss(**kwargs):
        try:
            input_data = json.loads(kwargs["input"])
            well_names_input: list[str] = input_data.get("wells")
            wells_dir = Naming.well_path()
            well_names = [f.name for f in os.scandir(wells_dir) if f.is_dir()]
            if well_names_input is not None and len(well_names_input) > 0:
                well_names = [f for f in well_names if f in well_names_input]
            well_names.sort()
            count = len(well_names)
            if count == 0:
                raise Exception(f"No wells found for {well_names_input}")

            elevation_path = Naming.elevation_file()
            elevation_kb: pd.Series | None = None
            elevation_well: pd.Series | None = None
            ELEVATION_WELL_COL = 2
            ELEVATION_KB_COL = 5
            if os.path.exists(elevation_path):
                elevation_df = pd.read_excel(elevation_path, header=1)
                elevation_well = elevation_df[
                    elevation_df.columns[ELEVATION_WELL_COL]
                ].astype(str)
                elevation_kb = elevation_df[elevation_df.columns[ELEVATION_KB_COL]]

            successCount = 0
            print(well_names)
            for well in well_names:
                print(well)
                devi_dir = Naming.devi_path(well)
                if not os.path.exists(devi_dir):
                    continue
                file_paths = [
                    f.path
                    for f in os.scandir(devi_dir)
                    if f.is_file() and f.name.lower().endswith(".txt")
                ]
                if len(file_paths) == 0:
                    continue
                file_path = file_paths[0]
                inp_df = pd.read_csv(file_path, sep="\\s+")
                DEPTH = inp_df.columns[0]
                AZIM = inp_df.columns[1]
                INCL = inp_df.columns[2]
                depth = inp_df[DEPTH].values
                azim = inp_df[AZIM].values
                incl = inp_df[INCL].values
                kb: float | None = None
                if elevation_kb is not None:
                    try:
                        kb = elevation_kb[elevation_well == well].values[0]
                    except:
                        pass

                tvd = [0] * len(depth)
                tvdss: list[float | None] = [None] * len(depth)
                for i in range(0, len(depth)):
                    if i == 0:
                        if kb is not None:
                            tvdss[i] = tvd[i] - kb
                        continue
                    dl = np.arccos(
                        np.cos(np.radians(incl[i - 1] - incl[i]))
                        - (
                            np.sin(np.radians(incl[i]))
                            * np.sin(np.radians(incl[i - 1]))
                            * (1 - np.cos(np.radians(azim[i - 1] - azim[i])))
                        )
                    )
                    rf = 1 if dl == 0 else ((2 / dl) * np.tan(dl / 2))
                    deltaTvd = (
                        (depth[i] - depth[i - 1])
                        * (
                            (
                                np.cos(np.radians(incl[i]))
                                + np.cos(np.radians(incl[i - 1]))
                            )
                            / 2
                        )
                        * rf
                    )
                    tvd[i] = tvd[i - 1] + deltaTvd
                    if kb is not None:
                        tvdss[i] = tvd[i] - kb

                df = pd.DataFrame(
                    data={
                        "DEPTH": depth,
                        "TVD": tvd,
                        "TVDSS": tvdss,
                    }
                )
                df.to_csv(Naming.tvdss_file(well), index=False, sep="\t")
                successCount += 1

            if successCount == 0:
                raise Exception(f"No wells created TVDss")

            return {"text": "Done"}
        except Exception as e:
            traceback.print_exc()
            return {"text": f"Tool failed: {str(e)}"}

    @mcp_server.tool(
        name="create_pseudo_log",
        description="Generate a curve for a well from curves in a list of wells using a machine learning model with model parameters"
    )
    def create_pseudo_log(**kwargs):
        try:
            input_data = json.loads(kwargs["input"])
            target_curve: str = input_data.get("target_curve")
            target_well: str = input_data.get("target_well")
            curves: list[str] = input_data.get("curves")
            wells: list[str] = input_data.get("wells")
            model_type: str = input_data.get("model_type")
            model_params: dict = input_data.get("model_params")
            wells_dir = Naming.well_path()

            available_wells = [entry.name for entry in os.scandir(wells_dir) if entry.is_dir()]
            selected_wells = [name for name in available_wells if name in wells] if wells else []
            selected_wells.sort()

            if not selected_wells:
                raise Exception(f"No valid wells found")
            
            if target_well not in available_wells:
                raise Exception(f"Well '{target_well}' does not exist")
            
            if not curves:
                raise Exception(f"No valid curves found")
                    
            started_event = Event()
            
            # start training
            process = Process(
                target=make_pseudo_log,
                args=(
                    target_curve, 
                    target_well, 
                    curves, 
                    selected_wells, 
                    model_type, 
                    model_params, 
                    started_event,
                )
            )
            process.start()

            # wait for the training process to init
            started_event.wait()
            
            # view training result
            out_file_relative_path = get_training_result(target_curve, target_well, model_type)

            return {"text": iframe(out_file_relative_path)}
        except Exception as e:
            traceback.print_exc()
            return {"text": f"Tool failed: {str(e)}"}

    @mcp_server.tool(
        name="view_training_experiment",
        description="Show the training experiments"
    )
    def view_training_experiment(**kwargs):
        try:
            input_data = json.loads(kwargs["input"])
            target_curve: str = input_data.get("target_curve")
            target_well: str = input_data.get("target_well")
            model_type: str = input_data.get("model_type")
            seconds: int = input_data.get("seconds")
            filter_expr: str = input_data.get("filter_expr")

            out_file_relative_path = get_training_result(target_curve, target_well, model_type, seconds, filter_expr)

            return {"text": iframe(out_file_relative_path, height='500px')}
        except Exception as e:
            traceback.print_exc()
            return {"text": f"Tool failed: {str(e)}"}
    
    @mcp_server.tool(
        name="delete_training_experiment",
        description="Delete the training experiment with experiment_id"
    )
    def delete_training_experiment(**kwargs):
        try:
            input_data = json.loads(kwargs["input"])
            experiment_id: str = input_data.get("experiment_id")

            # delete
            remove_training_result(experiment_id)
            
            # view
            out_file_relative_path = get_training_result()

            return {"text": iframe(out_file_relative_path, height='500px')}
        except Exception as e:
            traceback.print_exc()
            return {"text": f"Tool failed: {str(e)}"}

    @mcp_server.tool(
        name="zone4well",
        description="Create zones for well",
    )
    def zone4well(input):
        try:
            input_data = json.loads(input)
            well = input_data.get("well")
            file_path = input_data.get("file_path", Naming.default_marker_file())
            print(file_path)
            print(input_data)
            keyZoneDF, zoneDF = XLSX.extract_zones1(well, file_path=file_path)
            storage = Store()
            storage.save(keyZoneDF, Naming.keyzonename(well))
            storage.save(zoneDF, Naming.zonename(well))
            return dict(text=json.dumps(zoneDF.to_dict("records")))
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))

    @mcp_server.tool(
        name="suggest_log_creation",
        description="discover data and suggest how to calculate a log curve"
    )
    def suggest_log_creation(input):
        try:
            input_data = json.loads(input)
            target_curve = input_data['target_curve'].upper()
            target_well = input_data['target_well']
            wells = input_data['wells']
            if not target_curve:
                raise Exception("Please provide specific target curve to calculate")
            if not target_well:
                raise Exception("Please provide specific target well for calculation")

            log_rule = getLogRules(target_curve)
            print('----->', log_rule)
            if log_rule is None:
                raise Exception(f"Don't know how to create log curve {target_curve}")
            list_curve = [target_curve] + log_rule

            def calc_distance(row):
                df = XLSX.extract_wellpos([target_well, row['well']])
                if len(df.index) < 2:
                    return None 
                wells = df[['X', 'Y']].values
                return np.sqrt(np.sum(np.square(wells[0] - wells[1])))
                
            def calc_score(row):
                length = len(list_curve)
                score = 0
                for idx,c in enumerate(list_curve) :
                    score += (length - idx) * row[c]
                return score

            all_curves = get_curves_in_well(target_well)
            missing_input_curves = {}
            for c in log_rule:
                if len(find_similar_curves(c, all_curves)):
                    missing_input_curves[c] = 0
                else:
                    missing_input_curves[c] = 1
            target_well_input_features = set(log_rule)
            target_well_missing_input_features = { c for c in missing_input_curves if missing_input_curves[c] >0 }

            well_infos = None
            if not wells:
                well_infos = get_wells_has_curve(target_curve)
            else:
                well_infos = [ { 'well': w, "all_curves": get_curves_in_well(w) } for w in wells ]

            for winfo in well_infos:
                for c in list_curve:
                    if len(find_similar_curves(c, winfo['all_curves'])):
                        winfo[c] = 1
                    else: 
                        winfo[c] = 0


            df = pd.DataFrame(well_infos)
            selected_cols = ['well', target_curve] + log_rule
            df = df[ selected_cols ]
            df['score'] = df.apply(calc_score, axis=1)
            df['distance'] = df.apply(calc_distance, axis=1)
            df_score = df.sort_values(by=['score', 'distance'], ascending=[False, True])
            df = df.sort_values(by=['distance', 'score'], ascending=[True, False])
            dest_path = Naming.dest_path(f"{target_curve}_{target_well}", category='suggestion', format = 'xlsx')
            publish_path = Naming.publish_path(f"{target_curve}_{target_well}", category='suggestion', format = 'xlsx')
            XLSX.save_dataframe(df, dest_path)

            top5 = df.head(5)
            print('top5', top5)
            opt1_wells = list(top5['well'].values)
            opt1_top5_most_distance = top5['distance'].max()
            opt1_input_features = [c for c in log_rule if top5[c].sum() == 5]
            opt1_input_missing = [c for c in log_rule if top5[c].sum() < 5]
            
            top5 = df_score.head(5)
            print('----------\n','top5', top5)
            opt2_wells = list(top5['well'].values)
            opt2_top5_most_distance = top5['distance'].max()
            opt2_input_features = [c for c in log_rule if top5[c].sum() == 5]
            opt2_input_missing = [c for c in log_rule if top5[c].sum() < 5]
            # conclude
            print(set(list_curve), set(missing_input_curves), set(list_curve) - set(missing_input_curves))
            answer = f'''
###  Analysis on target well {target_well}:

1. Important input curves available:

- {'\n- '.join(list(target_well_input_features))}

2. The following important input curves are missing: {','.join(target_well_missing_input_features) or None}

### For calculating _{target_curve}_ in well _{target_well}_ consider the following suggestions:

1. Using nearby wells: {",".join(opt1_wells)} with input curves {",".join(opt1_input_features)} 
(missing: {",".join(opt1_input_missing) or 'None'}). 
The most distant well is {opt1_top5_most_distance} metres away from {target_well}.
Also consider reconstructing missing curves before calculating {target_curve}
2. Using wells with most available data: {",".join(opt2_wells)} with input curves {",".join(opt2_input_features)}
(missing: {",".join(opt2_input_missing) or 'None'}). 
The most distant well is {opt2_top5_most_distance} metres away from {target_well}.

The above conclusions are drawn from {excel_link(publish_path, label="here")}
'''
            return dict(text=answer)
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))

    @mcp_server.tool(
        name="accept_experiment_las_file",
        description="Accept las file produced by an experiment"
    )
    def accept_experiment_las_file(input):
        try:
            run_id_prefix = input
            runs = get_runs(run_id_prefix)
            if runs is None or len(runs) == 0:
                raise Exception("No experiment run found")
            run = runs[0]
            run_data = run.data
            well = run_data.params.get("target_well", None)
            las_file = os.path.basename(run_data.params.get('las_file'))
            if well is None:
                raise Exception('This experiment is not for well log')
            dest_path = Naming.las_path(well, category='store')
            print(las_file)
            download_artifacts(
                run_id=run.info.run_id, 
                artifact_path=f'las/{las_file}', 
                dst_path = f'{dest_path}/')
            file_path = f"{dest_path}/_{las_file}"
            os.rename(f"{dest_path}/{las_file}", file_path)
            las = lasio.read(file_path)
            storage = Store()
            new_curves = [{'curve': c.mnemonic, 'path': file_path, 'ref': None} for c in las.curves]
            storage.save_curves_in_well(new_curves, well)
            return dict(text=f"Experiment {run_id_prefix} result is saved to well {well} under file name _{las_file}")
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))

    @mcp_server.tool(
        name="plt_table",
        description="Build PLT Table"
    )
    def plt_table(input):
        try:
            input_data = input
            print(input_data)
            raw_path = 'plt.xlsx'
            dest_path = Naming.dest_path(raw_path, category='plt', format='xlsx')
            publish_path = Naming.publish_path(raw_path, category='plt', format='xlsx')
            if not os.path.exists(dest_path):
                in_file = Naming.default_plt_file(category='store')
                df = XLSX.parse_plt()
                XLSX.save_dataframe(df, dest_path)
            return dict(text=excel_link(publish_path, label="PLT Table"))
        except Exception as e:
            traceback.print_exc()
            return dict(text=str(e))
    
    tool_names = [
        "build_logplot",
        "zone4well",
        "well_checklist_table",
        "well_checklist_curves",
        "create_wells_tvdss",
        "create_pseudo_log",
        "view_training_experiment",
        "delete_training_experiment",
        "suggest_log_creation",
        "accept_experiment_las_file",
        "plt_table"
    ]
    return tool_names
