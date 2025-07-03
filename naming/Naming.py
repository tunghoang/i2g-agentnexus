class Naming:
    @classmethod
    def markername(cls, well):
        return f"well{well.strip()}.marker.csv"

    @classmethod
    def productionRecordName(cls, well):
        return f"well{well.strip()}.prodrecord.csv"

    @classmethod
    def histogramName(cls, lasname):
        return f"{lasname}.histogram.html"
