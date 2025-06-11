# production_segy_analysis.py - Use existing production tools
from production_segy_analysis_qc import production_segy_analysis
import json


def analyze_with_production_tools(file_path):
    """Use production SEG-Y tools that auto-handle templates"""

    print(f"🏭 Production SEG-Y Analysis: {file_path}")

    # The production tools automatically create templates if needed
    result = production_segy_analysis(
        file_path=file_path,
        analysis_type="full",  # Full analysis
        data_dir="./data",
        template_dir="./templates"
    )

    if "text" in result:
        data = json.loads(result["text"])

        if "error" not in data:
            print("SUCCESS!")
            print(f"Survey Type: {data.get('survey_type', 'Unknown')}")
            print(f"Total Traces: {data.get('total_traces', 0):,}")
            print(f"File Size: {data.get('file_size_mb', 0)} MB")
            print(f"Processing Time: {data.get('processing_time_seconds', 0)} seconds")

            if "geometry_analysis" in data:
                geo = data["geometry_analysis"]
                print(f"Geometry: {geo.get('survey_type', 'Unknown')}")

            return data
        else:
            print(f"Error: {data['error']}")
            return None

    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python production_segy_analysis.py <segy_file>")
        sys.exit(1)

    analyze_with_production_tools(sys.argv[1])