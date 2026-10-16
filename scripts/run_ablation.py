import argparse
import os
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_config
from app.core.logging_config import logger
from app.research.ablation import run_ablation_study

def main():
    parser = argparse.ArgumentParser(description="RoadSense AI — Run 5-Stage Ablation Experiment")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input smartphone CSV dataset")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to custom config.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    config = load_config(args.config)
    df_raw = pd.read_csv(args.input)

    logger.info(f"Running Ablation Study on {args.input} ({len(df_raw)} rows)...")
    df_results, results_list = run_ablation_study(df_raw, config=config)

    print("\n================ ABLATION STUDY EXPERIMENT RESULTS ================")
    print(df_results.to_string(index=False))
    print("====================================================================\n")

if __name__ == "__main__":
    main()
