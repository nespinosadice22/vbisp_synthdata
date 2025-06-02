import json, itertools, statistics
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# List of all patient IDs
PATIENT_IDS = [1023, 1024, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1044, 1045, 1046, 1047, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1079, 1080, 1081, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1103, 1104, 1105, 1106, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1128, 1129, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1143, 1144, 1145, 1146, 1148, 1149, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1163, 1164, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1188, 1189, 1192, 1193, 1194, 1195, 1196, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1297, 1298, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1359, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1372, 1373, 1374, 1376, 1377, 1378, 1379, 1380, 1381, 1383, 1384, 1385, 4009, 4019, 4022, 4026, 4028, 4030, 4033, 4034, 4035, 4037, 4038, 4041, 4042, 4044, 4046, 4051, 4054, 4058, 4066, 4077, 4088, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4122, 4123, 4125, 4127, 4128, 4130, 4131, 4133, 4134, 4135, 4136, 4140, 4141, 4142, 4143, 4145, 4146, 4148, 4149, 4150, 4153, 4154, 4155, 4158, 4159, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4175, 4179, 4181, 4182, 4183, 4184, 4185, 4186, 4188, 4190, 4191, 4193, 4196, 4200, 4201, 4205, 4206, 4208, 4210, 4212, 4216, 4219, 4220, 4221, 4228, 4230, 4234, 4235, 4236, 4237, 4240, 4241, 4248, 4255, 4257, 4263, 4268, 4273, 4281, 4282, 4283, 4284, 4285, 4286, 4289, 4291, 4298, 4301, 7025, 7037, 7038, 7039, 7040, 7041, 7043, 7044, 7045, 7047, 7048, 7049, 7051, 7052, 7053, 7056, 7058, 7059, 7061, 7062, 7063, 7064, 7065, 7066, 7067, 7068, 7069, 7070, 7071, 7072, 7073, 7074, 7076, 7077, 7078, 7079, 7080, 7081, 7082, 7084, 7086, 7087, 7089, 7090, 7092, 7093, 7096, 7097, 7098, 7099, 7100, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109, 7110, 7111, 7112, 7113, 7114, 7115, 7116, 7117, 7118, 7119, 7120, 7122, 7123, 7124, 7125, 7126, 7127, 7128, 7129, 7130, 7131, 7132, 7133, 7134, 7136, 7137, 7138, 7139, 7140, 7141, 7142, 7143, 7144, 7145, 7146, 7147, 7148, 7149, 7150, 7152, 7153, 7154, 7155, 7156, 7157, 7158, 7159, 7160, 7161, 7162, 7164, 7165, 7166, 7167, 7168, 7169, 7170, 7171, 7172, 7173, 7174, 7175, 7176, 7177, 7178, 7179, 7180, 7181, 7182, 7183, 7184, 7185, 7186, 7188, 7189, 7190, 7191, 7192, 7193, 7194, 7195, 7196, 7197, 7198, 7199, 7200, 7201, 7202, 7203, 7204, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213, 7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224, 7225, 7226, 7227, 7228, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238, 7239, 7240, 7241, 7242, 7243, 7244, 7245, 7246, 7247, 7248, 7249, 7250, 7251, 7252, 7253, 7254, 7255, 7256, 7257, 7258, 7259, 7260, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 7268, 7269, 7270, 7271, 7272, 7273, 7274, 7275, 7276, 7277, 7278, 7279, 7280, 7281, 7282, 7283, 7284, 7285, 7286, 7287, 7288, 7290, 7291, 7292, 7293, 7294, 7295, 7296, 7297, 7298, 7299, 7300, 7301, 7302, 7303, 7304, 7305, 7306, 7307, 7308, 7309, 7310, 7311, 7312, 7313, 7314, 7315, 7316, 7317, 7318, 7319, 7320, 7322, 7323, 7325, 7326, 7327, 7328, 7329, 7330, 7332, 7333, 7334, 7335, 7336, 7337, 7338, 7339, 7340, 7341, 7343, 7344, 7345, 7346, 7347, 7348, 7349, 7350, 7351, 7352, 7354, 7355, 7356, 7357, 7358, 7360, 7361, 7362, 7363, 7364, 7365, 7366, 7367, 7368, 7369, 7371, 7372, 7373, 7374, 7375, 7376, 7377, 7378, 7379, 7381, 7382, 7383, 7384, 7385, 7386, 7387, 7388, 7389, 7390, 7391, 7392, 7393, 7394, 7395, 7396, 7397, 7398, 7399, 7401, 7402, 7403, 7404, 7405, 7406, 7407, 7408, 7409, 7411]

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

#---------------------- Data loading functions ------------------------------- # 
def get_bloodglucose(path): 
    try:
        path = Path(path)
        with path.open() as f: 
            data = json.load(f) 
        times = [] 
        values = [] 
        for rec in data["body"]["cgm"]: 
            start_time = (rec["effective_time_frame"]["time_interval"]["start_date_time"])
            end_time = (rec["effective_time_frame"]["time_interval"]["end_date_time"])
            if start_time != end_time: 
                continue  # Skip inconsistent times
            v = rec["blood_glucose"]["value"]
            if v == 'Low': 
                v = float(-1.0)
            elif v == 'High': 
                v = float(-1.0)
            else: 
                v = float(rec["blood_glucose"]["value"])
            times.append(start_time)
            values.append(v) 
        return pd.DataFrame({"start_time": times, "blood_glucose": values}).sort_values(by='start_time')
    except Exception as e:
        return pd.DataFrame()

def get_heartrate(path): 
    try:
        path = Path(path)
        with path.open() as f: 
            data = json.load(f) 
        times = [] 
        values = [] 
        for rec in data["body"]["heart_rate"]: 
            t = (rec["effective_time_frame"]["date_time"])
            v = float(rec["heart_rate"]["value"])
            times.append(t)
            values.append(v) 
        return pd.DataFrame({"date_time": times, "heart_rate": values}).sort_values(by='date_time')
    except Exception as e:
        return pd.DataFrame()

def get_oxygensat(path): 
    try:
        path = Path(path)
        with path.open() as f: 
            data = json.load(f) 
        times = [] 
        values = [] 
        for rec in data["body"]["breathing"]: 
            t = (rec["effective_time_frame"]["date_time"])
            v = float(rec["oxygen_saturation"]["value"])
            times.append(t)
            values.append(v) 
        return pd.DataFrame({"date_time": times, "oxygen_saturation": values}).sort_values(by='date_time')
    except Exception as e:
        return pd.DataFrame()

def get_activity(path): 
    try:
        path = Path(path)
        with path.open() as f: 
            data = json.load(f) 
        start_times = [] 
        end_times = [] 
        act_names = [] 
        act_values = [] 
        act_units = [] 
        for rec in data["body"]["activity"]: 
            start_time = (rec["effective_time_frame"]["time_interval"]["start_date_time"])
            end_time = (rec["effective_time_frame"]["time_interval"]["end_date_time"])
            act_name = rec["activity_name"]
            act_value = rec["base_movement_quantity"]["value"]
            act_unit = rec["base_movement_quantity"]["unit"]
            start_times.append(start_time) 
            end_times.append(end_time)
            act_names.append(act_name)
            act_values.append(act_value)
            act_units.append(act_unit)
        return pd.DataFrame({"start_time": start_times, "end_time": end_times, "activity_name": act_names, "activity_value": act_values, "activity_units": act_units}).sort_values(by='start_time')
    except Exception as e:
        return pd.DataFrame()

def get_calorie(path): 
    try:
        path = Path(path)
        with path.open() as f: 
            data = json.load(f) 
        times = [] 
        values = [] 
        for rec in data["body"]["activity"]: 
            t = (rec["effective_time_frame"]["date_time"])
            v = rec["calories_value"]["value"]
            times.append(t)
            values.append(v) 
        return pd.DataFrame({"date_time": times, "calories": values}).sort_values(by='date_time')
    except Exception as e:
        return pd.DataFrame()

def get_respiratoryrate(path): 
    try:
        path = Path(path)
        with path.open() as f: 
            data = json.load(f) 
        times = [] 
        values = [] 
        for rec in data["body"]["breathing"]: 
            t = (rec["effective_time_frame"]["date_time"])
            v = rec["respiratory_rate"]["value"]
            times.append(t)
            values.append(v) 
        return pd.DataFrame({"date_time": times, "respiratory_rate": values}).sort_values(by='date_time')
    except Exception as e:
        return pd.DataFrame()

def get_sleep(path): 
    try:
        path = Path(path)
        with path.open() as f: 
            data = json.load(f) 
        start_times = [] 
        end_times = [] 
        values = [] 
        for rec in data["body"]["sleep"]: 
            start_time = (rec["sleep_stage_time_frame"]["time_interval"]["start_date_time"])
            end_time = (rec["sleep_stage_time_frame"]["time_interval"]["end_date_time"])
            v = rec["sleep_stage_state"]
            start_times.append(start_time)
            end_times.append(end_time)
            values.append(v) 
        return pd.DataFrame({"start_time": start_times, "end_times": end_times, "sleep_stage": values}).sort_values(by='start_time')
    except Exception as e:
        return pd.DataFrame()

def get_stress(path): 
    try:
        path = Path(path)
        with path.open() as f: 
            data = json.load(f) 
        times = [] 
        values = [] 
        for rec in data["body"]["stress"]: 
            t = (rec["effective_time_frame"]["date_time"])
            v = rec["stress"]["value"]
            times.append(t)
            values.append(v) 
        return pd.DataFrame({"date_time": times, "stress": values}).sort_values(by='date_time')
    except Exception as e:
        return pd.DataFrame()

def get_patient_data(patient_id): 
    """Load data for a single patient"""
    root = Path("~/Desktop/synth_data/dataset").expanduser()
    patient = str(patient_id)
    
    files = {
        "blood_glucose"  : root / "wearable_blood_glucose"    / "continuous_glucose_monitoring" / "dexcom_g6" / patient / f"{patient}_DEX.json",
        "heart_rate"     : root / "wearable_activity_monitor" / "heart_rate" / "garmin_vivosmart5" / patient / f"{patient}_heartrate.json",
        "oxygen_sat"     : root / "wearable_activity_monitor" / "oxygen_saturation" / "garmin_vivosmart5" / patient / f"{patient}_oxygensaturation.json",
        "activity"       : root / "wearable_activity_monitor" / "physical_activity" / "garmin_vivosmart5" / patient / f"{patient}_activity.json",
        "calorie"        : root / "wearable_activity_monitor" / "physical_activity_calorie" / "garmin_vivosmart5" / patient / f"{patient}_calorie.json",
        "resp_rate"      : root / "wearable_activity_monitor" / "respiratory_rate" / "garmin_vivosmart5" / patient / f"{patient}_respiratoryrate.json",
        "sleep"          : root / "wearable_activity_monitor" / "sleep" / "garmin_vivosmart5" / patient / f"{patient}_sleep.json",
        "stress"         : root / "wearable_activity_monitor" / "stress" / "garmin_vivosmart5" / patient / f"{patient}_stress.json",
    }
    
    patient_dataframes = {
        'blood_glucose': get_bloodglucose(files["blood_glucose"]),
        'heart_rate': get_heartrate(files["heart_rate"]),
        'oxygen_sat': get_oxygensat(files["oxygen_sat"]),
        'activity': get_activity(files["activity"]),
        'calorie': get_calorie(files["calorie"]),
        'resp_rate': get_respiratoryrate(files["resp_rate"]),
        'sleep': get_sleep(files["sleep"]),
        'stress': get_stress(files["stress"])
    }
    
    return patient_id, patient_dataframes

def calculate_sensor_intervals(df, time_col, sensor_name, patient_id):
    """Calculate intervals for a single sensor from a single patient"""
    if df.empty:
        return None
    
    try:
        # Convert to datetime and sort
        df_clean = df.copy()
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])
        df_clean = df_clean.sort_values(time_col).reset_index(drop=True)
        
        # Calculate time differences
        time_diffs = df_clean[time_col].diff().dropna()
        intervals_seconds = time_diffs.dt.total_seconds()
        
        if len(intervals_seconds) == 0:
            return None
            
        # Calculate statistics
        stats = {
            'patient_id': patient_id,
            'sensor': sensor_name,
            'n_measurements': len(df_clean),
            'n_intervals': len(intervals_seconds),
            'min_interval_sec': intervals_seconds.min(),
            'max_interval_sec': intervals_seconds.max(),
            'mean_interval_sec': intervals_seconds.mean(),
            'median_interval_sec': intervals_seconds.median(),
            'std_interval_sec': intervals_seconds.std(),
            'duration_hours': (df_clean[time_col].max() - df_clean[time_col].min()).total_seconds() / 3600,
            'measurement_rate_per_hour': len(df_clean) / max((df_clean[time_col].max() - df_clean[time_col].min()).total_seconds() / 3600, 0.1),
            'intervals_raw': intervals_seconds.values  # Store raw intervals for aggregation
        }
        
        return stats
        
    except Exception as e:
        print(f"Error processing {sensor_name} for patient {patient_id}: {e}")
        return None

def analyze_patient_intervals(patient_id):
    """Analyze intervals for all sensors for a single patient"""
    try:
        patient_id, patient_data = get_patient_data(patient_id)
        
        sensor_configs = [
            ('blood_glucose', 'start_time'),
            ('heart_rate', 'date_time'),
            ('oxygen_sat', 'date_time'),
            ('calorie', 'date_time'),
            ('resp_rate', 'date_time'),
            ('stress', 'date_time'),
        ]
        
        results = []
        
        for sensor_key, time_col in sensor_configs:
            if sensor_key in patient_data and not patient_data[sensor_key].empty:
                stats = calculate_sensor_intervals(
                    patient_data[sensor_key], 
                    time_col, 
                    sensor_key, 
                    patient_id
                )
                if stats:
                    results.append(stats)
        
        return results
        
    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        return []

def create_population_interval_plots(all_results, save_dir):
    """Create comprehensive plots showing interval patterns across all patients"""
    
    # Convert results to DataFrame for easier analysis
    df_results = pd.DataFrame([r for r in all_results if r is not None])
    
    if df_results.empty:
        print("No valid results to plot")
        return
    
    # Create summary statistics by sensor
    sensor_summary = df_results.groupby('sensor').agg({
        'n_measurements': ['count', 'mean', 'std', 'min', 'max'],
        'mean_interval_sec': ['mean', 'std', 'min', 'max'],
        'median_interval_sec': ['mean', 'std', 'min', 'max'],
        'measurement_rate_per_hour': ['mean', 'std', 'min', 'max'],
        'duration_hours': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    print("\n" + "="*80)
    print("POPULATION-LEVEL SENSOR INTERVAL ANALYSIS")
    print("="*80)
    
    for sensor in df_results['sensor'].unique():
        sensor_data = df_results[df_results['sensor'] == sensor]
        print(f"\n{sensor.upper().replace('_', ' ')} POPULATION SUMMARY:")
        print(f"  Patients with data: {len(sensor_data)}")
        print(f"  Measurements per patient: {sensor_data['n_measurements'].mean():.0f} ± {sensor_data['n_measurements'].std():.0f}")
        print(f"  Mean interval: {sensor_data['mean_interval_sec'].mean():.1f} ± {sensor_data['mean_interval_sec'].std():.1f} seconds")
        print(f"  Median interval: {sensor_data['median_interval_sec'].mean():.1f} ± {sensor_data['median_interval_sec'].std():.1f} seconds")
        print(f"  Measurement rate: {sensor_data['measurement_rate_per_hour'].mean():.1f} ± {sensor_data['measurement_rate_per_hour'].std():.1f} per hour")
        print(f"  Recording duration: {sensor_data['duration_hours'].mean():.1f} ± {sensor_data['duration_hours'].std():.1f} hours")
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(20, 16))
    
    sensors = df_results['sensor'].unique()
    n_sensors = len(sensors)
    
    # Plot 1: Mean interval distributions by sensor
    plt.subplot(3, 3, 1)
    for i, sensor in enumerate(sensors):
        sensor_data = df_results[df_results['sensor'] == sensor]
        plt.hist(sensor_data['mean_interval_sec']/60, alpha=0.6, bins=20, 
                label=f'{sensor.replace("_", " ").title()} (n={len(sensor_data)})')
    plt.xlabel('Mean Interval (minutes)')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Mean Intervals Across Patients')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Measurement rates by sensor
    plt.subplot(3, 3, 2)
    sensor_names = [s.replace('_', ' ').title() for s in sensors]
    rates_data = [df_results[df_results['sensor'] == s]['measurement_rate_per_hour'].values for s in sensors]
    plt.boxplot(rates_data, labels=sensor_names)
    plt.ylabel('Measurements per Hour')
    plt.title('Measurement Rates by Sensor Type')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Number of measurements per patient by sensor
    plt.subplot(3, 3, 3)
    measurements_data = [df_results[df_results['sensor'] == s]['n_measurements'].values for s in sensors]
    plt.boxplot(measurements_data, labels=sensor_names)
    plt.ylabel('Number of Measurements')
    plt.title('Measurements per Patient by Sensor')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Recording duration by sensor
    plt.subplot(3, 3, 4)
    duration_data = [df_results[df_results['sensor'] == s]['duration_hours'].values for s in sensors]
    plt.boxplot(duration_data, labels=sensor_names)
    plt.ylabel('Recording Duration (hours)')
    plt.title('Recording Duration by Sensor')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Interval variability (coefficient of variation)
    plt.subplot(3, 3, 5)
    cv_data = []
    for sensor in sensors:
        sensor_data = df_results[df_results['sensor'] == sensor]
        cv_values = sensor_data['std_interval_sec'] / sensor_data['mean_interval_sec']
        cv_data.append(cv_values.values)
    plt.boxplot(cv_data, labels=sensor_names)
    plt.ylabel('Coefficient of Variation')
    plt.title('Interval Variability by Sensor')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{save_dir}/population_interval_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved population analysis: {filename}")
    
    # Create separate figure for individual sensor histograms (all sensors)
    create_individual_sensor_histograms(df_results, save_dir)
    
    # Create sensor comparison heatmap
    create_sensor_comparison_heatmap(df_results, save_dir)
    
    # Create detailed sensor tables
    create_sensor_summary_table(df_results, save_dir)
    
    return df_results

def create_individual_sensor_histograms(df_results, save_dir):
    """Create individual histogram plots for all sensors"""
    
    sensors = df_results['sensor'].unique()
    n_sensors = len(sensors)
    
    # Calculate grid size for all sensors
    n_cols = 3
    n_rows = (n_sensors + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array
    
    for i, sensor in enumerate(sensors):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Combine all intervals from all patients for this sensor
        all_intervals = []
        sensor_data = df_results[df_results['sensor'] == sensor]
        for _, row_data in sensor_data.iterrows():
            if 'intervals_raw' in row_data and row_data['intervals_raw'] is not None:
                all_intervals.extend(row_data['intervals_raw'])
        
        if all_intervals:
            intervals_minutes = np.array(all_intervals) / 60
            
            # Use log scale for wide range of intervals
            if len(intervals_minutes) > 0 and intervals_minutes.max() / intervals_minutes.min() > 100:
                bins = np.logspace(np.log10(max(intervals_minutes.min(), 0.1)), 
                                 np.log10(intervals_minutes.max()), 50)
                ax.set_xscale('log')
            else:
                bins = 50
            
            ax.hist(intervals_minutes, bins=bins, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Interval (minutes)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{sensor.replace("_", " ").title()} Intervals\n(All Patients Combined)')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            median_int = np.median(intervals_minutes)
            mean_int = np.mean(intervals_minutes)
            ax.axvline(median_int, color='red', linestyle='--', alpha=0.8, 
                      label=f'Median: {median_int:.1f}min')
            ax.axvline(mean_int, color='orange', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_int:.1f}min')
            ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for\n{sensor.replace("_", " ").title()}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{sensor.replace("_", " ").title()} Intervals')
    
    # Hide unused subplots
    for i in range(n_sensors, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    filename = f"{save_dir}/individual_sensor_histograms.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved individual sensor histograms: {filename}")

def create_sensor_comparison_heatmap(df_results, save_dir):
    """Create heatmap comparing sensors across different metrics"""
    
    # Calculate summary statistics for heatmap
    sensors = df_results['sensor'].unique()
    metrics = ['mean_interval_sec', 'median_interval_sec', 'measurement_rate_per_hour', 
              'n_measurements', 'duration_hours']
    
    heatmap_data = []
    for sensor in sensors:
        sensor_data = df_results[df_results['sensor'] == sensor]
        row = []
        for metric in metrics:
            row.append(sensor_data[metric].mean())
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=[s.replace('_', ' ').title() for s in sensors],
                             columns=['Mean Interval (s)', 'Median Interval (s)', 
                                    'Rate (per hr)', 'N Measurements', 'Duration (hrs)'])
    
    # Normalize for better visualization
    heatmap_df_norm = heatmap_df.div(heatmap_df.max(axis=0), axis=1)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_df_norm, annot=True, fmt='.2f', cmap='viridis', 
                cbar_kws={'label': 'Normalized Value'})
    plt.title('Sensor Characteristics Comparison\n(Normalized to 0-1 scale)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{save_dir}/sensor_comparison_heatmap.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sensor comparison heatmap: {filename}")

def create_sensor_summary_table(df_results, save_dir):
    """Create and save detailed summary tables"""
    
    # Overall summary table
    summary_stats = []
    for sensor in df_results['sensor'].unique():
        sensor_data = df_results[df_results['sensor'] == sensor]
        
        stats = {
            'Sensor': sensor.replace('_', ' ').title(),
            'N_Patients': len(sensor_data),
            'Avg_Measurements': f"{sensor_data['n_measurements'].mean():.0f} ± {sensor_data['n_measurements'].std():.0f}",
            'Avg_Duration_Hours': f"{sensor_data['duration_hours'].mean():.1f} ± {sensor_data['duration_hours'].std():.1f}",
            'Avg_Rate_per_Hour': f"{sensor_data['measurement_rate_per_hour'].mean():.1f} ± {sensor_data['measurement_rate_per_hour'].std():.1f}",
            'Mean_Interval_Sec': f"{sensor_data['mean_interval_sec'].mean():.1f} ± {sensor_data['mean_interval_sec'].std():.1f}",
            'Median_Interval_Sec': f"{sensor_data['median_interval_sec'].mean():.1f} ± {sensor_data['median_interval_sec'].std():.1f}",
            'Min_Interval_Sec': f"{sensor_data['min_interval_sec'].mean():.1f}",
            'Max_Interval_Sec': f"{sensor_data['max_interval_sec'].mean():.0f}",
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save to CSV
    summary_filename = f"{save_dir}/population_sensor_summary.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"Saved summary table: {summary_filename}")
    
    # Print formatted table
    print("\n" + "="*120)
    print("DETAILED SENSOR SUMMARY TABLE")
    print("="*120)
    print(summary_df.to_string(index=False))
    print("="*120)

def main():
    """Main function to run population-level interval analysis"""
    
    print("MULTI-PATIENT SENSOR INTERVAL ANALYSIS")
    print("="*80)
    print(f"Analyzing {len(PATIENT_IDS)} patients...")
    
    # Create output directory
    save_dir = "./population_analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    # Use multiprocessing for faster analysis
    print("Processing patients in parallel...")
    
    all_results = []
    
    # Process in chunks to manage memory
    chunk_size = 50
    
    for i in range(0, len(PATIENT_IDS), chunk_size):
        chunk = PATIENT_IDS[i:i+chunk_size]
        print(f"Processing patients {i+1}-{min(i+chunk_size, len(PATIENT_IDS))}...")
        
        with ProcessPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
            future_to_patient = {executor.submit(analyze_patient_intervals, patient_id): patient_id 
                               for patient_id in chunk}
            
            for future in tqdm(as_completed(future_to_patient), total=len(chunk)):
                patient_id = future_to_patient[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error processing patient {patient_id}: {e}")
    
    print(f"\nSuccessfully processed {len(all_results)} sensor datasets")
    
    # Track which patients have blood glucose data
    patients_with_bg = set()
    sensor_counts = {}
    
    for result in all_results:
        if result:
            sensor = result.get('sensor', 'unknown')
            patient_id = result.get('patient_id')
            
            if sensor not in sensor_counts:
                sensor_counts[sensor] = 0
            sensor_counts[sensor] += 1
            
            if sensor == 'blood_glucose':
                patients_with_bg.add(patient_id)
    
    # Generate missing blood glucose patients analysis
    missing_bg_patients = [pid for pid in PATIENT_IDS if pid not in patients_with_bg]
    
    print(f"\nData availability summary:")
    total_patients = len(PATIENT_IDS)
    for sensor, count in sensor_counts.items():
        missing = total_patients - count
        print(f"  {sensor}: {count}/{total_patients} patients ({missing} missing/empty)")
    
    # Special detailed report for blood glucose
    print(f"\n" + "="*60)
    print(f"BLOOD GLUCOSE MISSING PATIENTS ANALYSIS")
    print(f"="*60)
    print(f"Total patients analyzed: {total_patients}")
    print(f"Patients with blood glucose data: {len(patients_with_bg)}")
    print(f"Patients missing blood glucose data: {len(missing_bg_patients)}")
    
    if missing_bg_patients:
        print(f"\nMissing blood glucose patients (first 50):")
        for i, patient_id in enumerate(missing_bg_patients[:50]):
            print(f"  {patient_id}", end="")
            if (i + 1) % 10 == 0:  # New line every 10 patients
                print()
        if len(missing_bg_patients) > 50:
            print(f"\n  ... and {len(missing_bg_patients) - 50} more")
        else:
            print()  # Final newline
        
        # Check why some patients are missing - analyze a few examples
        print(f"\nAnalyzing reasons for missing data (first 5 patients):")
        root = Path("~/Desktop/synth_data/dataset").expanduser()
        
        for patient_id in missing_bg_patients[:5]:
            patient = str(patient_id)
            bg_file = root / "wearable_blood_glucose" / "continuous_glucose_monitoring" / "dexcom_g6" / patient / f"{patient}_DEX.json"
            
            if not bg_file.exists():
                print(f"  Patient {patient_id}: File does not exist - {bg_file}")
            elif bg_file.stat().st_size == 0:
                print(f"  Patient {patient_id}: File exists but is empty")
            else:
                try:
                    df = get_bloodglucose(bg_file)
                    if df.empty:
                        print(f"  Patient {patient_id}: File exists but no valid data after parsing")
                    else:
                        print(f"  Patient {patient_id}: File has {len(df)} records (unexpected - should have data)")
                except Exception as e:
                    print(f"  Patient {patient_id}: Error reading file - {e}")
    
    # Save the missing patients list to file
    missing_file = f"{save_dir}/missing_blood_glucose_patients.txt"
    with open(missing_file, 'w') as f:
        f.write(f"Missing Blood Glucose Patients ({len(missing_bg_patients)} total)\n")
        f.write("="*50 + "\n\n")
        for patient_id in missing_bg_patients:
            f.write(f"{patient_id}\n")
    print(f"\nSaved complete list to: {missing_file}")
    
    print(f"\nCreating population-level visualizations...")

    # Create comprehensive analysis and plots
    df_results = create_population_interval_plots(all_results, save_dir)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"Results saved to: {save_dir}/")
    print(f"Total datasets analyzed: {len(all_results)}")
    print(f"{'='*80}")
    
    return df_results

if __name__ == "__main__":
    results = main()