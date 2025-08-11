from pathlib import Path
import os, shutil


def ensure_path(inpath):
    path_to_ensure = inpath
    if inpath.endswith((".html", ".json", ".js", ".css", ".xlsx")):
        path_to_ensure = os.path.dirname(inpath)
    Path(path_to_ensure).mkdir(parents=True, exist_ok=True)


class Naming:
    @classmethod
    def context_path(cls):
        return 'context.json'
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename by keeping only alphanumeric characters and safe symbols."""
        ALLOWED_CHARS = set(".-_")
        return "".join(c for c in filename if c.isalnum() or c in ALLOWED_CHARS)

    @classmethod
    def markername(cls, well):
        return f"well{well.strip()}.marker.csv"

    @classmethod
    def zonename(cls, well):
        return f"well{well.strip()}.zone.csv"

    @classmethod
    def keyzonename(cls, well):
        return f"well{well.strip()}.keyzone.csv"

    @classmethod
    def productionRecordName(cls, well):
        return f"well{well.strip()}.prodrecord.csv"

    @classmethod
    def histogramName(cls, lasname):
        return f"{lasname}.histogram.html"

    @classmethod
    def dest_path(cls, inpath, category="", format='html', create_dir=True):
        CHART_DIR = "/tmp"
        outpath = f"{CHART_DIR}/{category}/{inpath}" if category else f"{CHART_DIR}/{inpath}"
        if format:
            outpath = outpath + f".{format}"
        if create_dir:
            ensure_path(outpath)
        return outpath

    @classmethod
    def publish_path(cls, inpath, category="", format="html"):
        outpath = f"{category}/{inpath}" if category else f"{inpath}"
        if format:
            outpath = f"{outpath}.{format}"
        return outpath

    @classmethod
    def data_path(cls, inpath, prefix="./data"):
        return f"{prefix}/{inpath}"

    @classmethod
    def __path_classifier(cls, default_path, category="raw", format=None):
        if category == "store":
            return cls.data_path(default_path)
        elif category == "raw":
            return default_path
        elif category == "publish":
            return cls.publish_path(default_path)
        elif category == 'temp':
            return cls.dest_path(default_path, format=format)
        else:
            return cls.data_path(default_path)
        
    @classmethod
    def default_marker_file(cls, category="store"):
        default_path = "misc/Marker.xlsx"
        return cls.__path_classifier(default_path, category=category)

    @classmethod
    def default_perforation_file(cls, category="store"):
        default_path = "misc/perforation.xlsx"
        return cls.__path_classifier(default_path, category = category)

    @classmethod
    def default_production_monthly_file(cls, category='raw'):
        #default_path = "production/monthly_production_data.xlsx"
        default_path = "production/PVT_WellTest_Perforation_WaterAnalysis.xlsx"
        return cls.__path_classifier(default_path, category = category)

    @classmethod
    def elevation_file(cls, category='store'):
        default_path = "misc/elevation.xlsx"
        return cls.__path_classifier(default_path, category = category)

    @classmethod
    def well_path(cls, well: str | None = None):
        if well is None:
            return cls.data_path("wells")
        return cls.data_path(f"wells/{well}")

    @classmethod
    def devi_path(cls, well: str):
        return f"{cls.well_path(well)}/GIS/Devi"
    
    @classmethod
    def las_path(cls, well: str):
        return f"{cls.well_path(well)}/GIS/Las"

    @classmethod
    def tvdss_file(cls, well: str):
        return f"{cls.devi_path(well)}/TVDSS.csv"

    @classmethod
    def gen_site(cls):
        files_to_gen = {
            'excel-viewer': 1, # directory and its subdirs recursively
            'js/plotly-3.0.1.min': 'js',
            'js/plotly-utils': 'js',
            'view_plot': 'html'
        }
        for fpath,ext in files_to_gen.items():
            if type(ext) == str:
                destPath = Naming.dest_path(fpath, format=ext)
                if not os.path.exists(destPath):
                    ensure_path(destPath)
                    shutil.copyfile(f'templates/{fpath}.{ext}', destPath)
            elif ext == 1:
                destPath = Naming.dest_path(fpath, format='', create_dir=False)
                print(destPath)
                if not os.path.isdir(destPath):
                    ensure_path(destPath)
                    shutil.rmtree(destPath)
                    print('Copytree', 'public/excel-viewer', destPath)
                    shutil.copytree('public/excel-viewer', destPath)
