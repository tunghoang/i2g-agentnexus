from pathlib import Path
import os
def ensure_path(inpath):
    path_to_ensure = inpath
    if inpath.endswith('.html'):
        path_to_ensure = os.path.dirname(inpath)
    Path(path_to_ensure).mkdir(parents=True, exist_ok=True)

class Naming:
    @classmethod
    def markername(cls, well):
        return f"well{well.strip()}.marker.csv"
    @classmethod
    def zonename(cls, well):
        return f"well{well.strip()}.zone.csv"

    @classmethod
    def productionRecordName(cls, well):
        return f"well{well.strip()}.prodrecord.csv"

    @classmethod
    def histogramName(cls, lasname):
        return f"{lasname}.histogram.html"

    @classmethod
    def dest_path(cls, inpath, category=""):
        CHART_DIR = "/tmp"
        outpath = f"{CHART_DIR}/{category}/{inpath}.html" if category else f"{CHART_DIR}/{category}/{inpath}.html"
        ensure_path(outpath)
        return outpath

    @classmethod
    def publish_path(cls, inpath, category=""):
        outpath = f"{category}/{inpath}.html" if category else f"{inpath}.html"
        return outpath

    @classmethod
    def data_path(cls, inpath, prefix='./data'):
        return f'{prefix}/{inpath}'

    @classmethod
    def default_marker_file(cls, category='store'):
        if category == 'store':
            return cls.data_path('misc/Marker.xlsx')
        elif category == 'raw':
            return 'misc/Marker.xlsx'
        elif category == 'publish':
            return cls.publish_path('misc/Marker.xlsx')
