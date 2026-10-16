================================================================================
  ROADSENSE AI — README.txt
  Vehicle-Independent Road Anomaly Classification Through
  DSP-Based Signal Calibration Across Diverse Vehicle and Road Conditions
================================================================================

PROJECT TITLE
-------------
  Vehicle-Independent Road Anomaly Classification Through DSP-Based Signal
  Calibration Across Diverse Vehicle and Road Conditions

REVIEW STATUS
-------------
  Review 2 — Implementation Progress (October 2026)

TEAM MEMBERS
------------
  Name              Roll Number
  ──────────────────────────────────
  Rahul Bodduna     2520040001
  Guna Shekar       2520040100
  Pavan Biradhar    2520040102

SUPERVISOR
----------
  Dr. V. Sahiti

INSTITUTION
-----------
  Department of Computer Science and Engineering
  [Your College Name] — [City], India


================================================================================
  1. SOFTWARE REQUIREMENTS
================================================================================

  Operating System  : Windows 10 / 11, Ubuntu 20.04+, or macOS 12+
  Python Version    : Python 3.11 (recommended); Python 3.9+ is compatible
  Package Manager   : pip (comes with Python)

  Core Libraries (see requirements.txt for version-pinned list):
  ┌─────────────────────┬────────────────────────────────────────────────┐
  │ Library             │ Purpose                                        │
  ├─────────────────────┼────────────────────────────────────────────────┤
  │ fastapi >=0.110.0   │ REST API web framework                         │
  │ uvicorn >=0.28.0    │ ASGI server to run FastAPI                     │
  │ pydantic >=2.6.0    │ Data validation and settings management        │
  │ pandas >=2.2.0      │ CSV I/O, data manipulation                     │
  │ numpy >=1.26.0      │ Numerical operations (FFT, arrays)             │
  │ scipy >=1.12.0      │ Butterworth filter, signal processing          │
  │ scikit-learn >=1.4.0│ Random Forest classifier, metrics              │
  │ matplotlib >=3.8.0  │ Figure generation (DSP plots, confusion matrix)│
  │ plotly >=5.19.0     │ Interactive charts in the web API              │
  │ folium >=0.16.0     │ GPS-based Leaflet map generation (HTML)        │
  │ joblib >=1.3.2      │ Model serialization (.joblib file)             │
  │ pyyaml >=6.0.1      │ Reading config.yaml                            │
  │ pytest >=8.0.0      │ Unit and integration tests                     │
  │ requests >=2.31.0   │ HTTP client for API testing                    │
  └─────────────────────┴────────────────────────────────────────────────┘

  Optional (for presenting results):
  - python-pptx >= 0.6.23   (only needed if regenerating the Review 2 PPT)


================================================================================
  2. INSTALLATION INSTRUCTIONS
================================================================================

  Step 1: Install Python 3.11
  ───────────────────────────
    Download from: https://www.python.org/downloads/release/python-3110/
    During install, CHECK "Add Python to PATH".

  Step 2: Open a Terminal / Command Prompt
  ─────────────────────────────────────────
    Windows : Press Win+R → type "cmd" or "powershell" → Enter
    Linux/macOS : Open terminal

  Step 3: Navigate to the project folder
  ───────────────────────────────────────
    cd path\to\roadsense-ai

  Step 4: (Recommended) Create a Virtual Environment
  ────────────────────────────────────────────────────
    python -m venv venv
    
    Activate it:
      Windows  : venv\Scripts\activate
      Linux/Mac: source venv/bin/activate

  Step 5: Install all required libraries
  ───────────────────────────────────────
    pip install -r requirements.txt

  Step 6: Verify installation
  ───────────────────────────
    python -c "import fastapi, pandas, sklearn, folium; print('OK')"
    → Should print: OK


================================================================================
  3. PROJECT FOLDER STRUCTURE AND FILE DESCRIPTIONS
================================================================================

  roadsense-ai/
  │
  ├── README.txt                  ← THIS FILE (setup and execution guide)
  ├── config.yaml                 ← Central configuration for all DSP and ML parameters
  ├── requirements.txt            ← Python package dependencies
  ├── .gitignore                  ← Files and folders excluded from version control
  │
  ├── app/                        ← Main application source code (Python package)
  │   ├── main.py                 ← FastAPI app entry point; registers all API routers
  │   │
  │   ├── api/                    ← HTTP API endpoint definitions
  │   │   ├── dataset.py          │  Input: CSV file upload  Output: validation report (JSON)
  │   │   ├── dsp.py              │  Input: validated CSV    Output: DSP-processed data (JSON)
  │   │   ├── features.py         │  Input: DSP data         Output: feature vectors (JSON)
  │   │   ├── ml.py               │  Input: feature vectors  Output: ML predictions (JSON)
  │   │   ├── predictions.py      │  Input: none             Output: saved predictions (JSON)
  │   │   ├── map.py              │  Input: predictions + GPS Output: HTML map path (JSON)
  │   │   └── research.py         │  Input: all data         Output: ablation/cross-vehicle (JSON)
  │   │
  │   ├── core/                   ← Shared utilities
  │   │   ├── config.py           │  Loads config.yaml; provides get_project_root()
  │   │   ├── logging_config.py   │  Configures structured console logger
  │   │   └── exceptions.py       │  Custom exception classes for error handling
  │   │
  │   ├── dsp/                    ← Digital Signal Processing pipeline
  │   │   ├── pipeline.py         │  Orchestrates all DSP stages; INPUT: raw DataFrame OUTPUT: processed DataFrame
  │   │   ├── validation.py       │  Checks CSV columns, timestamps, GPS validity
  │   │   ├── sampling.py         │  Estimates and resamples to target Hz (default 50 Hz)
  │   │   ├── hampel.py           │  Hampel identifier for outlier/spike removal (window=7, 3-sigma)
  │   │   ├── gravity.py          │  Low-pass Butterworth filter to separate gravity component
  │   │   ├── filtering.py        │  Band-pass Butterworth filter (0.7–15 Hz) for road vibrations
  │   │   ├── magnitude.py        │  Computes vector magnitude of accelerometer and gyroscope
  │   │   ├── jerk.py             │  Computes jerk (dAcc/dt in m/s³) for impact detection
  │   │   ├── calibration.py      │  Median-MAD normalization to remove vehicle-specific bias
  │   │   └── anomaly_detection.py│  Detects high-jerk events; labels POTHOLE vs SPEED BREAKER
  │   │
  │   ├── features/               ← Feature extraction
  │   │   ├── extractor.py        │  Orchestrates windowed feature extraction; INPUT: DSP DataFrame OUTPUT: feature DataFrame
  │   │   ├── time_domain.py      │  Computes mean, std, RMS, peak, kurtosis, skewness per window
  │   │   ├── frequency_domain.py │  FFT-based features: dominant frequency, spectral energy, rolloff
  │   │   └── sensor_fusion.py    │  Cross-sensor correlation and combined accelerometer+gyroscope features
  │   │
  │   ├── ml/                     ← Machine learning models and evaluation
  │   │   ├── dataset_builder.py  │  Prepares (X, y, groups) for session-aware training
  │   │   ├── trainer.py          │  Trains Random Forest (300 trees, balanced class weight)
  │   │   ├── evaluator.py        │  Session-Aware GroupShuffleSplit evaluation; outputs accuracy, F1, confusion matrix
  │   │   ├── predictor.py        │  Loads trained model and predicts on new feature windows
  │   │   └── model_registry.py   │  Saves/loads model + label encoder + metadata as .joblib file
  │   │
  │   ├── mapping/                ← GPS and map generation
  │   │   ├── gps.py              │  Filters GPS rows by accuracy and drops missing coordinates
  │   │   ├── ground_truth.py     │  Extracts manual/labeled anomaly ground truth from raw CSV
  │   │   ├── road_segmentation.py│  Classifies 15-second road windows as SMOOTH or ROUGH
  │   │   └── route.py            │  Generates Folium interactive HTML map with markers and layer control
  │   │
  │   └── research/               ← Research experiment modules
  │       ├── ablation.py         │  5-stage DSP ablation study (raw → validated → filtered → calibrated → all)
  │       ├── cross_vehicle.py    │  3 cross-vehicle generalization experiments (train 2+3 test 4, etc.)
  │       ├── metrics.py          │  Generates all research figures (confusion matrix, class dist, FFT, jerk, etc.)
  │       └── report.py           │  Writes experiment_summary.md with full metrics and tables
  │
  ├── scripts/                    ← Standalone execution scripts (run from project root)
  │   ├── run_pipeline.py         ← MAIN SCRIPT: Full end-to-end pipeline (DSP → ML → Map → Research)
  │   ├── train.py                ← Train the model only (skips DSP; reads features.csv)
  │   ├── evaluate.py             ← Evaluate the saved model on existing features
  │   ├── run_ablation.py         ← Run only the DSP ablation study
  │   ├── run_cross_vehicle.py    ← Run only the cross-vehicle generalization experiments
  │   └── create_notebooks.py     ← Creates skeleton Jupyter notebooks in notebooks/
  │
  ├── data/                       ← All data files
  │   ├── raw/                    ← USER-PROVIDED input CSV files (place your data here)
  │   │   ├── data1.csv           │  User dataset 1 (uploaded for testing)
  │   │   ├── data2.csv           │  User dataset 2 (uploaded for testing)
  │   │   ├── dataset.csv         │  Combined/reference dataset
  │   │   └── RoadSense_ride_01_2026-09-27.csv  │  Sample ride session CSV
  │   │
  │   ├── processed/              ← Auto-generated after running the pipeline
  │   │   ├── 2W_processed.csv    │  DSP-processed 2-Wheeler (Bike) data
  │   │   ├── 3W_processed.csv    │  DSP-processed 3-Wheeler (Auto-rickshaw) data
  │   │   ├── 4W_processed.csv    │  DSP-processed 4-Wheeler (Car) data
  │   │   ├── dsp_processed.csv   │  Combined DSP output (all vehicles)
  │   │   └── road_segments.csv   │  Road classification by 15-second segments
  │   │
  │   ├── features/               ← Auto-generated feature vectors
  │   │   └── features.csv        │  Time-domain + frequency-domain features per window
  │   │
  │   └── predictions/            ← Auto-generated ML output
  │       └── predictions.csv     │  Predicted road class and confidence per window
  │
  ├── models/                     ← Saved trained model
  │   └── random_forest_model.joblib  ← Serialized Random Forest + LabelEncoder + metadata
  │
  ├── results/                    ← All experimental outputs
  │   ├── experiment_summary.md   │  Full research report: metrics, ablation table, cross-vehicle table
  │   ├── ablation_results.csv    │  Per-stage accuracy and F1 scores for ablation study
  │   ├── cross_vehicle_results.csv│ Cross-vehicle experiment results (Exp A, B, C)
  │   ├── road_map.html           │  Interactive GPS map (open in any browser)
  │   └── figures/                ← Generated research plots (PNG)
  │       ├── raw_vs_filtered.png         │  Raw signal vs Butterworth-filtered signal
  │       ├── filtered_vs_calibrated.png  │  Filtered vs Median-MAD calibrated signal
  │       ├── fft_analysis.png            │  FFT frequency spectrum of the signal
  │       ├── jerk_analysis.png           │  Numerical jerk (dAcc/dt) over time
  │       ├── impact_score.png            │  Anomaly impact intensity score timeline
  │       ├── class_distribution.png      │  Distribution of road condition labels
  │       └── confusion_matrix.png        │  ML confusion matrix (POTHOLE/SMOOTH/SPEED BREAKER)
  │
  ├── notebooks/                  ← Jupyter notebooks for analysis and exploration
  │   ├── 01_dataset_exploration.ipynb    │  Load and explore raw CSV data
  │   ├── 02_dsp_analysis.ipynb           │  Visualize DSP stages step-by-step
  │   ├── 03_feature_analysis.ipynb       │  Inspect extracted features
  │   ├── 04_ml_training.ipynb            │  Train and evaluate the Random Forest model
  │   ├── 05_ablation_study.ipynb         │  Run and visualize DSP ablation experiments
  │   └── 06_cross_vehicle_evaluation.ipynb│ Run and analyze cross-vehicle experiments
  │
  ├── tests/                      ← Automated unit and integration tests
  │   ├── conftest.py             │  Shared pytest fixtures (sample DataFrames, config, etc.)
  │   └── test_pipeline.py        │  Tests for DSP pipeline, feature extraction, and ML flow
  │
  └── RoadSense_AI_Review_2_Implementation_Progress.pptx  ← Review 2 Presentation


================================================================================
  4. INPUT CSV FILE FORMAT (Required Columns)
================================================================================

  Your input CSV must contain at minimum these columns:

  Column Name      Type      Description
  ──────────────────────────────────────────────────────────────────────────────
  timestamp        datetime  ISO 8601 or Unix epoch (e.g., 2026-09-27 10:30:00)
  acc_x            float     Accelerometer X-axis (m/s²)
  acc_y            float     Accelerometer Y-axis (m/s²)
  acc_z            float     Accelerometer Z-axis (m/s²)
  gyro_x           float     Gyroscope X-axis (rad/s)   [optional but recommended]
  gyro_y           float     Gyroscope Y-axis (rad/s)   [optional but recommended]
  gyro_z           float     Gyroscope Z-axis (rad/s)   [optional but recommended]
  latitude         float     GPS latitude (decimal degrees)
  longitude        float     GPS longitude (decimal degrees)
  gps_accuracy     float     GPS accuracy in metres (lower = better)
  label            string    Road condition label: SMOOTH / ROUGH / POTHOLE / SPEED BREAKER
                             (can be empty if ground truth is unknown)
  vehicle_type     string    2-WHEELER / 3-WHEELER / 4-WHEELER  [optional — auto-detected]
  session_id       string    Unique ride session ID (used for GroupShuffleSplit)

  For Vehicle-Specific Files:
    - 2-Wheeler (Bike)       : name the file 2W_Bike.csv or include "2w" in filename
    - 3-Wheeler (Auto)       : name the file 3W_Auto.csv or include "3w"/"auto" in filename
    - 4-Wheeler (Car)        : name the file 4W_Car.csv or include "4w"/"car" in filename

  DATASET SOURCES (External):
  ────────────────────────────
  This project uses self-collected smartphone sensor data. If you need public
  reference datasets for comparison, see:

  1. ROPME Dataset (Road and Pavement Monitoring for Mobile Environments):
     https://github.com/jmpr0/ROPME
     Description: Smartphone accelerometer data for road surface monitoring.

  2. IEEE DataPort — Road Surface Anomaly Detection:
     https://ieee-dataport.org/
     Search: "road anomaly detection smartphone accelerometer"

  3. PhysioNet Motion Sensor Data:
     https://physionet.org/
     Search: "accelerometer vibration transport"

  For self-collected data: collect data using any Android sensor logger app
  (e.g., Physics Toolbox, Sensor Logger) and export as CSV with the columns
  described above.


================================================================================
  5. CONFIGURING FILE PATHS AND SETTINGS
================================================================================

  All parameters are controlled in config.yaml (project root).
  You should NOT need to change source code for routine runs.

  KEY SETTINGS IN config.yaml:
  ──────────────────────────────
  sampling:
    target_hz: 50           # Target resampling rate. Change if your sensor runs at a different Hz.

  butterworth:
    low_cutoff_hz: 0.7      # Lower frequency bound for road vibration band-pass filter
    high_cutoff_hz: 15.0    # Upper frequency bound (noise above 15 Hz is suppressed)

  hampel:
    window_size: 7          # Sliding window for spike detection (samples)
    n_sigma: 3.0            # Outlier threshold (samples > 3 std deviations from median)

  window:
    duration_seconds: 2.0   # Length of each analysis window for feature extraction
    overlap_seconds: 0.0    # Overlap between consecutive windows (set > 0 for denser output)

  model:
    n_estimators: 300       # Number of trees in Random Forest
    random_state: 42        # Random seed for reproducibility

  paths:
    data_raw: data/raw                  # Folder for input CSV files
    data_processed: data/processed      # Auto-generated DSP output
    data_features: data/features        # Auto-generated feature vectors
    data_predictions: data/predictions  # Auto-generated ML predictions
    models: models                      # Saved model location
    results: results                    # All output files and figures

  IMPORTANT: All paths in config.yaml are relative to the PROJECT ROOT.
  Do NOT use absolute paths inside config.yaml.


================================================================================
  6. MAIN FILE TO RUN AND CORRECT EXECUTION SEQUENCE
================================================================================

  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  MAIN ENTRY POINT: scripts/run_pipeline.py                              ║
  ╚══════════════════════════════════════════════════════════════════════════╝

  Prerequisites:
    1. Python 3.11 installed.
    2. Virtual environment activated (see Section 2).
    3. Libraries installed: pip install -r requirements.txt
    4. Your CSV file(s) placed in the data/raw/ folder.

  ──────────────────────────────────────────────────────────────────────────
  OPTION A: Run the complete pipeline (RECOMMENDED)
  ──────────────────────────────────────────────────────────────────────────
    From the project root directory:

      python scripts/run_pipeline.py --input data/raw/data1.csv

    Flags:
      --input  (-i)  : Path to your primary CSV dataset (default: data/raw/hyderabad_dataset.csv)
      --config (-c)  : Path to a custom config.yaml (optional; uses root config.yaml by default)

    What happens automatically:
      Step 1  → Discovers 2W / 3W / 4W datasets in data/raw/
      Step 2  → Validates each CSV (checks columns, timestamps, GPS)
      Step 3  → DSP Processing (resampling, outlier removal, gravity removal,
                Butterworth filtering, magnitude, jerk, Median-MAD calibration)
      Step 4  → Feature Extraction (time-domain + FFT features per 2-second window)
      Step 5  → ML Training and Evaluation (Random Forest, GroupShuffleSplit)
      Step 6  → Prediction Generation (per-window road class + confidence)
      Step 7  → Cross-Vehicle Experiments (Exp A: train2W+3W test4W, etc.)
      Step 8  → DSP Ablation Study (5 progressive stages)
      Step 9  → Interactive GPS Map Generation (Folium HTML)
      Step 10 → Research Figures Generation (PNG plots)
      Step 11 → Experiment Summary Report (results/experiment_summary.md)

  ──────────────────────────────────────────────────────────────────────────
  OPTION B: Run the FastAPI web application (interactive UI)
  ──────────────────────────────────────────────────────────────────────────
    From the project root directory:

      python app/main.py

    Then open your browser and go to:
      http://127.0.0.1:8000        ← API status page
      http://127.0.0.1:8000/docs   ← Interactive Swagger UI (upload CSV, run DSP, view predictions)

    API Workflow via browser:
      POST /upload      → Upload your CSV file
      POST /dsp         → Run DSP processing on uploaded file
      POST /features    → Extract features from processed data
      POST /predict     → Run ML classification
      GET  /map         → View GPS map HTML

  ──────────────────────────────────────────────────────────────────────────
  OPTION C: Run individual steps separately
  ──────────────────────────────────────────────────────────────────────────
    # Train the model only (requires features.csv to exist):
      python scripts/train.py

    # Evaluate saved model:
      python scripts/evaluate.py

    # Run ablation study only:
      python scripts/run_ablation.py

    # Run cross-vehicle experiments only:
      python scripts/run_cross_vehicle.py

  ──────────────────────────────────────────────────────────────────────────
  OPTION D: Run automated tests
  ──────────────────────────────────────────────────────────────────────────
      pytest tests/ -v


================================================================================
  7. EXPECTED OUTPUTS AND WHERE RESULTS ARE SAVED
================================================================================

  After running scripts/run_pipeline.py, the following files are created:

  OUTPUT FILE                             LOCATION
  ──────────────────────────────────────────────────────────────────────────────
  2W_processed.csv                        data/processed/
  3W_processed.csv                        data/processed/
  4W_processed.csv                        data/processed/
  dsp_processed.csv (all vehicles)        data/processed/
  road_segments.csv                       data/processed/
  features.csv                            data/features/
  predictions.csv                         data/predictions/
  random_forest_model.joblib              models/
  road_map.html  ← Open in browser        results/
  experiment_summary.md                   results/
  ablation_results.csv                    results/
  cross_vehicle_results.csv              results/
  raw_vs_filtered.png                     results/figures/
  filtered_vs_calibrated.png             results/figures/
  fft_analysis.png                        results/figures/
  jerk_analysis.png                       results/figures/
  impact_score.png                        results/figures/
  class_distribution.png                 results/figures/
  confusion_matrix.png                    results/figures/

  Console Output Summary:
    The pipeline prints a table showing per-vehicle row counts, duration,
    sampling rate, and anomaly counts (Potholes, Speed Breakers).
    It also prints the full ablation and cross-vehicle evaluation tables.

  Viewing the GPS Map:
    Open results/road_map.html in any web browser (Chrome, Firefox, Edge).
    The map shows:
      - Green markers  = Smooth road sections
      - Orange markers = Speed Breakers
      - Red markers    = Potholes
      - Layer toggles  = Filter by vehicle type (2W / 3W / 4W)


================================================================================
  8. KNOWN LIMITATIONS AND EXECUTION ISSUES
================================================================================

  Issue 1: "ModuleNotFoundError" when running scripts
  ────────────────────────────────────────────────────
    Cause   : Running the script from outside the project root.
    Fix     : Always run from the project root folder:
                cd path\to\roadsense-ai
                python scripts/run_pipeline.py

  Issue 2: Pipeline runs with minimal data (no labels)
  ──────────────────────────────────────────────────────
    Cause   : Input CSV has no "label" column or all labels are empty.
    Effect  : ML training is skipped; the pre-saved model in models/ is used.
    Fix     : Either label your data with SMOOTH/ROUGH/POTHOLE/SPEED BREAKER,
              or ensure models/random_forest_model.joblib exists from a prior run.

  Issue 3: Only 1 vehicle dataset found — synthetic variations used
  ─────────────────────────────────────────────────────────────────
    Cause   : Only one CSV file is in data/raw/.
    Effect  : The pipeline creates synthetic 3W and 4W datasets from the 2W data
              (with slight noise variations) to enable cross-vehicle experiments.
    Note    : Cross-vehicle results from synthetic data should be treated as
              exploratory; real multi-vehicle results require separate real datasets.
    Fix     : Place actual 2W_Bike.csv, 3W_Auto.csv, and 4W_Car.csv in data/raw/.

  Issue 4: GPS map markers not showing
  ──────────────────────────────────────
    Cause   : GPS columns are missing, NaN, or accuracy values exceed 50 m.
    Fix     : Ensure latitude, longitude, and gps_accuracy columns are present
              and valid. Set gps_accuracy threshold in config.yaml → mapping →
              min_gps_accuracy_m.

  Issue 5: Sampling rate warnings
  ─────────────────────────────────
    Cause   : Input data has irregular timestamps or was collected at a rate
              different from config.yaml → sampling → target_hz (default: 50 Hz).
    Effect  : Data is automatically resampled to target_hz.
    Fix     : If your sensor runs at a known rate (e.g., 100 Hz), update
              config.yaml → sampling → target_hz accordingly.

  Issue 6: "POTHOLE" class not predicted
  ────────────────────────────────────────
    Cause   : Pothole events may be under-represented in the training data,
              or dataset labels are only SMOOTH and SPEED BREAKER.
    Fix     : Collect more pothole events; use class_weight: balanced in config.yaml.
              The confusion matrix in results/figures/confusion_matrix.png shows
              which classes are being confused.

  Issue 7: python-pptx not installed (PPT regeneration)
  ───────────────────────────────────────────────────────
    Cause   : The Review 2 PPT was generated using python-pptx which is not in
              the default requirements.txt.
    Fix     : pip install python-pptx
              The PPT file (RoadSense_AI_Review_2_Implementation_Progress.pptx) is
              already pre-generated and included in this submission folder.

  Issue 8: Long pipeline runtime
  ────────────────────────────────
    Cause   : Large CSV files or many cross-vehicle experiments.
    Fix     : Use a smaller test CSV initially. On a standard laptop (i5/8GB RAM),
              the pipeline on typical ride data (~50,000 rows) takes 1–3 minutes.

  RESEARCH INTEGRITY NOTE:
  ─────────────────────────
    The POTHOLE class is absent from the confusion matrix (confusion_matrix.png)
    in current results because the available labelled data does not contain
    confirmed pothole events. All reported accuracy figures apply to the classes
    that are actually present in the labelled training data. Do not infer POTHOLE
    classification accuracy from these results.


================================================================================
  9. UPLOAD TO KMS — CHECKLIST
================================================================================

  The ZIP file (RoadSense_AI_Review_2.zip) submitted to KMS contains:

    [x] README.txt                                (this file)
    [x] config.yaml                               (DSP and ML settings)
    [x] requirements.txt                          (Python dependencies)
    [x] app/                                      (all source code)
    [x] scripts/                                  (pipeline execution scripts)
    [x] data/raw/                                 (input CSV datasets)
    [x] data/processed/                           (DSP output)
    [x] data/features/                            (ML feature vectors)
    [x] data/predictions/                         (ML predictions)
    [x] models/random_forest_model.joblib         (trained model)
    [x] results/experiment_summary.md             (research report)
    [x] results/ablation_results.csv              (ablation study)
    [x] results/cross_vehicle_results.csv         (cross-vehicle study)
    [x] results/road_map.html                     (GPS visualization)
    [x] results/figures/*.png                     (all research plots)
    [x] notebooks/                                (Jupyter analysis notebooks)
    [x] tests/                                    (unit tests)
    [x] RoadSense_AI_Review_2_Implementation_Progress.pptx  (Review 2 PPT)

  NOT included in ZIP (excluded to reduce size):
    - __pycache__/ folders                        (auto-generated bytecode)
    - venv/ folder                                (virtual environment)
    - .git/ folder                                (version control history)


================================================================================
  10. QUICK START (COPY-PASTE COMMANDS)
================================================================================

  # 1. Install Python 3.11 from https://www.python.org

  # 2. Open terminal and go to project root
  cd path\to\roadsense-ai

  # 3. Create virtual environment
  python -m venv venv
  venv\Scripts\activate        (Windows)
  source venv/bin/activate     (Linux/Mac)

  # 4. Install dependencies
  pip install -r requirements.txt

  # 5. Run the full pipeline
  python scripts/run_pipeline.py --input data/raw/data1.csv

  # 6. Open the GPS map in browser
  start results/road_map.html          (Windows)
  open results/road_map.html           (macOS)
  xdg-open results/road_map.html       (Linux)

  # 7. (Optional) Launch the web API
  python app/main.py
  # Then open: http://127.0.0.1:8000/docs


================================================================================
  END OF README.txt
================================================================================
