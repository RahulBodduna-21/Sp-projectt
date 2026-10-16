import argparse
import glob
import os
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_config
from app.core.logging_config import logger
from app.research.cross_vehicle import run_cross_vehicle_study

def main():
    parser = argparse.ArgumentParser(description="RoadSense AI — Run Leave-One-Vehicle-Out Cross-Vehicle Experiment")
    parser.add_argument("--input", "-i", type=str, required=True, help="Directory containing vehicle subfolders (Auto, Bike, Car) or CSV files")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to custom config.yaml")
    args = parser.parse_args()

    input_dir = args.input
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    config = load_config(args.config)
    vehicle_datasets = {}

    # Check for subdirectories Auto, Bike, Car
    subdirs = ["Auto", "Bike", "Car"]
    for s in subdirs:
        s_path = os.path.join(input_dir, s)
        if os.path.isdir(s_path):
            csvs = glob.glob(os.path.join(s_path, "*.csv"))
            if csvs:
                logger.info(f"Loading {len(csvs)} CSV files for vehicle type: {s}")
                df_list = [pd.read_csv(c) for c in csvs]
                vehicle_datasets[s] = pd.concat(df_list, ignore_index=True)

    # If no subdirs, inspect CSV files directly for vehicle_type column or filename hints
    if not vehicle_datasets:
        csvs = glob.glob(os.path.join(input_dir, "*.csv")) + glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
        for c in csvs:
            fname = os.path.basename(c).lower()
            df = pd.read_csv(c)
            if "bike" in fname or ("vehicle_type" in df.columns and (df["vehicle_type"].astype(str).str.lower() == "bike").any()):
                vehicle_datasets["Bike"] = pd.concat([vehicle_datasets.get("Bike", pd.DataFrame()), df], ignore_index=True)
            elif "auto" in fname or ("vehicle_type" in df.columns and (df["vehicle_type"].astype(str).str.lower() == "auto").any()):
                vehicle_datasets["Auto"] = pd.concat([vehicle_datasets.get("Auto", pd.DataFrame()), df], ignore_index=True)
            elif "car" in fname or ("vehicle_type" in df.columns and (df["vehicle_type"].astype(str).str.lower() == "car").any()):
                vehicle_datasets["Car"] = pd.concat([vehicle_datasets.get("Car", pd.DataFrame()), df], ignore_index=True)

    df_results, res_dict = run_cross_vehicle_study(vehicle_datasets, config=config)

    print("\n================ CROSS-VEHICLE EXPERIMENT RESULTS ================")
    print(df_results.to_string(index=False))
    print("===================================================================\n")

if __name__ == "__main__":
    main()
