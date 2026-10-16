import pandas as pd
import numpy as np

def calculate_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates 3D vector magnitudes for acceleration and gyroscope signals."""
    df_mag = df.copy()
    
    # Acceleration magnitude
    acc_x = df_mag["acc_x"].values if "acc_x" in df_mag.columns else np.zeros(len(df_mag))
    acc_y = df_mag["acc_y"].values if "acc_y" in df_mag.columns else np.zeros(len(df_mag))
    acc_z = df_mag["acc_z"].values if "acc_z" in df_mag.columns else np.zeros(len(df_mag))
    
    df_mag["acc_magnitude"] = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Gyroscope magnitude
    gyro_x = df_mag["gyro_x"].values if "gyro_x" in df_mag.columns else np.zeros(len(df_mag))
    gyro_y = df_mag["gyro_y"].values if "gyro_y" in df_mag.columns else np.zeros(len(df_mag))
    gyro_z = df_mag["gyro_z"].values if "gyro_z" in df_mag.columns else np.zeros(len(df_mag))
    
    df_mag["gyro_magnitude"] = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    
    return df_mag
