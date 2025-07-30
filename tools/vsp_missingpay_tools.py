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
from utils.missing_pay_utils import get_well_checklist, get_well_checklist_curves
from utils.plot_utils import multi_chart, advLogplot, logplot
from xlsx_utils import XLSX


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

            well_data_dir = f"wells/{well}"
            if not os.path.isdir(f"{data_config.data_dir}/{well_data_dir}"):
                raise Exception(f"Well {well} not found")
            composite_logs = f"{well_data_dir}/GIS/Las/*.las"
            files = glob(f"{data_config.data_dir}/{composite_logs}")
            if len(files) == 0:
                raise Exception(f"No composite logs found for well {well}")

            track_templates = input_data.get("track_templates", "GR,LLD,NPHI")
            track_templates = track_templates.split(",")
            track_templates = [tpl.strip() for tpl in track_templates]

            logDF_path = files[0]

            las = MemoryCache.get_instance().get(logDF_path)
            if las is None:
                las = lasio.read(logDF_path)
                MemoryCache.get_instance().put(logDF_path, las)

            df = las.df().reset_index()

            if track_templates:
                keyZoneDF, zoneDF = XLSX.extract_zones1(well)
                fig = advLogplot(df, las.curves, track_styles=track_templates, title=f"Well {well} Logplot", keyZoneDF = keyZoneDF, zoneDF=zoneDF)
            else:
                fig = logplot(df, las.curves)

            dest_path = Naming.dest_path(
                logDF_path.removeprefix(f"{data_config.data_dir}/"), category="logplot"
            )
            fig.write_html(dest_path)
            return {
                "text": Naming.publish_path(
                    logDF_path.removeprefix(f"{data_config.data_dir}/"),
                    category="logplot",
                )
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

            return {"text": out_file_relative_path}
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
                f"well_checklist_curves{"_".join(well_names)}.html"
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
            for well in well_names:
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

    tool_names = [
        "build_logplot",
        "zone4well",
        "well_checklist_table",
        "well_checklist_curves",
        "create_wells_tvdss",
    ]
    return tool_names
