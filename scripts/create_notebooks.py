import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
nb_dir = ROOT / "notebooks"
os.makedirs(nb_dir, exist_ok=True)

notebooks = [
    ("01_dataset_exploration.ipynb", "Dataset Exploration and Validation Analysis"),
    ("02_dsp_analysis.ipynb", "DSP Signal Calibration and Frequency Filtering Analysis"),
    ("03_feature_analysis.ipynb", "Time and Frequency Domain Feature Extraction Analysis"),
    ("04_ml_training.ipynb", "Random Forest Model Training and Evaluation"),
    ("05_ablation_study.ipynb", "5-Stage Experimental Ablation Study"),
    ("06_cross_vehicle_evaluation.ipynb", "Leave-One-Vehicle-Out Cross-Vehicle Generalization"),
    ("07_final_results.ipynb", "Final Research Synthesis and Visualization")
]

for fname, title in notebooks:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# RoadSense AI — {title}\n\n",
                    "This research notebook evaluates actual pipeline outputs and outputs generated in `data/` and `results/`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys, os, pandas as pd, numpy as np\n",
                    "sys.path.insert(0, '..')\n",
                    "from app.core.config import load_config\n",
                    "config = load_config()\n",
                    "print('Loaded RoadSense AI configuration successfully.')\n"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(nb_dir / fname, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

print(f"Successfully generated {len(notebooks)} Jupyter notebooks in {nb_dir}")
