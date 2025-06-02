import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Patient IDs from your list
PATIENT_IDS = [1023, 1024, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1044, 1045, 1046, 1047, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1079, 1080, 1081, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1103, 1104, 1105, 1106, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1128, 1129, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1143, 1144, 1145, 1146, 1148, 1149, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1163, 1164, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1188, 1189, 1192, 1193, 1194, 1195, 1196, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1297, 1298, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1359, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1372, 1373, 1374, 1376, 1377, 1378, 1379, 1380, 1381, 1383, 1384, 1385, 4009, 4019, 4022, 4026, 4028, 4030, 4033, 4034, 4035, 4037, 4038, 4041, 4042, 4044, 4046, 4051, 4054, 4058, 4066, 4077, 4088, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4122, 4123, 4125, 4127, 4128, 4130, 4131, 4133, 4134, 4135, 4136, 4140, 4141, 4142, 4143, 4145, 4146, 4148, 4149, 4150, 4153, 4154, 4155, 4158, 4159, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4175, 4179, 4181, 4182, 4183, 4184, 4185, 4186, 4188, 4190, 4191, 4193, 4196, 4200, 4201, 4205, 4206, 4208, 4210, 4212, 4216, 4219, 4220, 4221, 4228, 4230, 4234, 4235, 4236, 4237, 4240, 4241, 4248, 4255, 4257, 4263, 4268, 4273, 4281, 4282, 4283, 4284, 4285, 4286, 4289, 4291, 4298, 4301, 7025, 7037, 7038, 7039, 7040, 7041, 7043, 7044, 7045, 7047, 7048, 7049, 7051, 7052, 7053, 7056, 7058, 7059, 7061, 7062, 7063, 7064, 7065, 7066, 7067, 7068, 7069, 7070, 7071, 7072, 7073, 7074, 7076, 7077, 7078, 7079, 7080, 7081, 7082, 7084, 7086, 7087, 7089, 7090, 7092, 7093, 7096, 7097, 7098, 7099, 7100, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109, 7110, 7111, 7112, 7113, 7114, 7115, 7116, 7117, 7118, 7119, 7120, 7122, 7123, 7124, 7125, 7126, 7127, 7128, 7129, 7130, 7131, 7132, 7133, 7134, 7136, 7137, 7138, 7139, 7140, 7141, 7142, 7143, 7144, 7145, 7146, 7147, 7148, 7149, 7150, 7152, 7153, 7154, 7155, 7156, 7157, 7158, 7159, 7160, 7161, 7162, 7164, 7165, 7166, 7167, 7168, 7169, 7170, 7171, 7172, 7173, 7174, 7175, 7176, 7177, 7178, 7179, 7180, 7181, 7182, 7183, 7184, 7185, 7186, 7188, 7189, 7190, 7191, 7192, 7193, 7194, 7195, 7196, 7197, 7198, 7199, 7200, 7201, 7202, 7203, 7204, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213, 7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224, 7225, 7226, 7227, 7228, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238, 7239, 7240, 7241, 7242, 7243, 7244, 7245, 7246, 7247, 7248, 7249, 7250, 7251, 7252, 7253, 7254, 7255, 7256, 7257, 7258, 7259, 7260, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 7268, 7269, 7270, 7271, 7272, 7273, 7274, 7275, 7276, 7277, 7278, 7279, 7280, 7281, 7282, 7283, 7284, 7285, 7286, 7287, 7288, 7290, 7291, 7292, 7293, 7294, 7295, 7296, 7297, 7298, 7299, 7300, 7301, 7302, 7303, 7304, 7305, 7306, 7307, 7308, 7309, 7310, 7311, 7312, 7313, 7314, 7315, 7316, 7317, 7318, 7319, 7320, 7322, 7323, 7325, 7326, 7327, 7328, 7329, 7330, 7332, 7333, 7334, 7335, 7336, 7337, 7338, 7339, 7340, 7341, 7343, 7344, 7345, 7346, 7347, 7348, 7349, 7350, 7351, 7352, 7354, 7355, 7356, 7357, 7358, 7360, 7361, 7362, 7363, 7364, 7365, 7366, 7367, 7368, 7369, 7371, 7372, 7373, 7374, 7375, 7376, 7377, 7378, 7379, 7381, 7382, 7383, 7384, 7385, 7386, 7387, 7388, 7389, 7390, 7391, 7392, 7393, 7394, 7395, 7396, 7397, 7398, 7399, 7401, 7402, 7403, 7404, 7405, 7406, 7407, 7408, 7409, 7411]

def get_activity_data(path):
    try:
        path = Path(path)
        if not path.exists():
            return None
        with path.open() as f:
            data = json.load(f)
        start_times = []
        end_times = []
        act_names = []
        for rec in data["body"]["activity"]:
            start_time = rec["effective_time_frame"]["time_interval"]["start_date_time"]
            end_time = rec["effective_time_frame"]["time_interval"]["end_date_time"]
            act_name = rec["activity_name"]
            start_times.append(start_time)
            end_times.append(end_time)
            act_names.append(act_name)
        return pd.DataFrame({"start_time": start_times, "end_time": end_times, "activity_name": act_names}).sort_values(by='start_time')
    except Exception as e:
        print(f"Error loading activity data from {path}: {e}")
        return None

def get_sleep_data(path):
    try:
        path = Path(path)
        if not path.exists():
            return None
        with path.open() as f:
            data = json.load(f)
        start_times = []
        end_times = []
        sleep_stages = []
        for rec in data["body"]["sleep"]:
            start_time = rec["sleep_stage_time_frame"]["time_interval"]["start_date_time"]
            end_time = rec["sleep_stage_time_frame"]["time_interval"]["end_date_time"]
            stage = rec["sleep_stage_state"]
            start_times.append(start_time)
            end_times.append(end_time)
            sleep_stages.append(stage)
        return pd.DataFrame({"start_time": start_times, "end_time": end_times, "sleep_stage": sleep_stages}).sort_values(by='start_time')
    except Exception as e:
        print(f"Error loading sleep data from {path}: {e}")
        return None

def analyze_continuity(df, data_type, patient_id):
    if df is None or df.empty:
        return {
            'patient_id': patient_id,
            'data_type': data_type,
            'has_data': False,
            'total_intervals': 0,
            'total_duration_hours': 0,
            'time_span_hours': 0,
            'coverage_percentage': 0,
            'max_gap_hours': 0,
            'gaps_over_1hr': 0,
            'gaps_over_6hr': 0,
            'gaps_over_24hr': 0,
            'longest_continuous_hours': 0
        }
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df = df.sort_values('start_time').reset_index(drop=True)
    total_intervals = len(df)
    first_time = df['start_time'].min()
    last_time = df['end_time'].max()
    time_span = last_time - first_time
    time_span_hours = time_span.total_seconds() / 3600
    
    #total duration covered by intervals
    df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 3600
    total_duration_hours = df['duration'].sum()
    coverage_percentage = (total_duration_hours / time_span_hours * 100) if time_span_hours > 0 else 0
    #gaps between intervals
    gaps = []
    continuous_periods = []
    current_continuous_start = df.iloc[0]['start_time']
    current_continuous_end = df.iloc[0]['end_time']
    for i in range(1, len(df)):
        prev_end = df.iloc[i-1]['end_time']
        curr_start = df.iloc[i]['start_time']
        gap_duration = (curr_start - prev_end).total_seconds() / 3600
        if gap_duration > 0.1:  
            gaps.append(gap_duration)
            continuous_periods.append((current_continuous_end - current_continuous_start).total_seconds() / 3600)
            current_continuous_start = curr_start
        current_continuous_end = max(current_continuous_end, df.iloc[i]['end_time'])
    continuous_periods.append((current_continuous_end - current_continuous_start).total_seconds() / 3600)
    max_gap_hours = max(gaps) if gaps else 0
    gaps_over_1hr = sum(1 for gap in gaps if gap > 1)
    gaps_over_6hr = sum(1 for gap in gaps if gap > 6)
    gaps_over_24hr = sum(1 for gap in gaps if gap > 24)
    longest_continuous_hours = max(continuous_periods) if continuous_periods else 0
    return {
        'patient_id': patient_id,
        'data_type': data_type,
        'has_data': True,
        'total_intervals': total_intervals,
        'total_duration_hours': round(total_duration_hours, 2),
        'time_span_hours': round(time_span_hours, 2),
        'coverage_percentage': round(coverage_percentage, 1),
        'max_gap_hours': round(max_gap_hours, 2),
        'gaps_over_1hr': gaps_over_1hr,
        'gaps_over_6hr': gaps_over_6hr,
        'gaps_over_24hr': gaps_over_24hr,
        'longest_continuous_hours': round(longest_continuous_hours, 2),
        'first_time': first_time,
        'last_time': last_time
    }

def analyze_all_patients(root_path="~/Desktop/synth_data/dataset", sample_size=None):
    root = Path(root_path).expanduser()
    patients_to_analyze = PATIENT_IDS[:sample_size] if sample_size else PATIENT_IDS
    results = []
    print(f"Analyzing {len(patients_to_analyze)} patients...")
    print("=" * 60)
    for i, patient_id in enumerate(patients_to_analyze):
        if i % 50 == 0:
            print(f"Processing patient {i+1}/{len(patients_to_analyze)}: {patient_id}")
        patient_str = str(patient_id)
        activity_path = root / "wearable_activity_monitor" / "physical_activity" / "garmin_vivosmart5" / patient_str / f"{patient_str}_activity.json"
        sleep_path = root / "wearable_activity_monitor" / "sleep" / "garmin_vivosmart5" / patient_str / f"{patient_str}_sleep.json"
        activity_df = get_activity_data(activity_path)
        activity_analysis = analyze_continuity(activity_df, 'activity', patient_id)
        results.append(activity_analysis)
        sleep_df = get_sleep_data(sleep_path)
        sleep_analysis = analyze_continuity(sleep_df, 'sleep', patient_id)
        results.append(sleep_analysis)
    return pd.DataFrame(results)

def create_summary_visualizations(results_df, save_dir="./continuity_analysis"):
    os.makedirs(save_dir, exist_ok=True)
    activity_results = results_df[results_df['data_type'] == 'activity'].copy()
    sleep_results = results_df[results_df['data_type'] == 'sleep'].copy()
    
    fig = plt.figure(figsize=(20, 16))
    
    #data availability overview
    plt.subplot(3, 3, 1)
    data_availability = results_df.groupby('data_type')['has_data'].agg(['sum', 'count'])
    data_availability['percentage'] = (data_availability['sum'] / data_availability['count']) * 100
    bars = plt.bar(data_availability.index, data_availability['percentage'], color=['skyblue', 'lightcoral'], alpha=0.8)
    plt.title('Data Availability by Type', fontweight='bold')
    plt.ylabel('Percentage of Patients with Data')
    plt.ylim(0, 100)
    for bar, pct in zip(bars, data_availability['percentage']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    #coverage percentage distribution
    plt.subplot(3, 3, 2)
    activity_with_data = activity_results[activity_results['has_data'] == True]
    sleep_with_data = sleep_results[sleep_results['has_data'] == True]
    if not activity_with_data.empty and not sleep_with_data.empty:
        plt.hist(activity_with_data['coverage_percentage'], bins=20, alpha=0.7, label='Activity', color='skyblue', edgecolor='black')
        plt.hist(sleep_with_data['coverage_percentage'], bins=20, alpha=0.7, label='Sleep', color='lightcoral', edgecolor='black')
        plt.xlabel('Coverage Percentage')
        plt.ylabel('Number of Patients')
        plt.title('Distribution of Coverage Percentages', fontweight='bold')
        plt.legend()
    
    #total duration distribution
    plt.subplot(3, 3, 3)
    if not activity_with_data.empty and not sleep_with_data.empty:
        plt.hist(activity_with_data['total_duration_hours'], bins=20, alpha=0.7, 
                label='Activity', color='skyblue', edgecolor='black')
        plt.hist(sleep_with_data['total_duration_hours'], bins=20, alpha=0.7, 
                label='Sleep', color='lightcoral', edgecolor='black')
        plt.xlabel('Total Duration (hours)')
        plt.ylabel('Number of Patients')
        plt.title('Distribution of Total Data Duration', fontweight='bold')
        plt.legend()
    
    #time span distribution  
    plt.subplot(3, 3, 4)
    if not activity_with_data.empty and not sleep_with_data.empty:
        plt.hist(activity_with_data['time_span_hours'], bins=20, alpha=0.7, 
                label='Activity', color='skyblue', edgecolor='black')
        plt.hist(sleep_with_data['time_span_hours'], bins=20, alpha=0.7, 
                label='Sleep', color='lightcoral', edgecolor='black')
        plt.xlabel('Time Span (hours)')
        plt.ylabel('Number of Patients')
        plt.title('Distribution of Data Time Spans', fontweight='bold')
        plt.legend()
    
    #Gap analysis
    plt.subplot(3, 3, 5)
    gap_types = ['gaps_over_1hr', 'gaps_over_6hr', 'gaps_over_24hr']
    gap_labels = ['> 1 hour', '> 6 hours', '> 24 hours']
    if not activity_with_data.empty and not sleep_with_data.empty:
        activity_gaps = [activity_with_data[gap].sum() for gap in gap_types]
        sleep_gaps = [sleep_with_data[gap].sum() for gap in gap_types]
        x = np.arange(len(gap_labels))
        width = 0.35
        plt.bar(x - width/2, activity_gaps, width, label='Activity', color='skyblue', alpha=0.8)
        plt.bar(x + width/2, sleep_gaps, width, label='Sleep', color='lightcoral', alpha=0.8)
        plt.xlabel('Gap Duration')
        plt.ylabel('Total Number of Gaps')
        plt.title('Gap Analysis Across All Patients', fontweight='bold')
        plt.xticks(x, gap_labels)
        plt.legend()
    #longest continuous periods
    plt.subplot(3, 3, 6)
    if not activity_with_data.empty and not sleep_with_data.empty:
        plt.scatter(activity_with_data['longest_continuous_hours'], range(len(activity_with_data)), alpha=0.6, label='Activity', color='skyblue')
        plt.scatter(sleep_with_data['longest_continuous_hours'], range(len(sleep_with_data)), alpha=0.6, label='Sleep', color='lightcoral')
        plt.xlabel('Longest Continuous Period (hours)')
        plt.ylabel('Patient Index')
        plt.title('Longest Continuous Periods per Patient', fontweight='bold')
        plt.legend()
    
    #correlation between activity and sleep coverage
    plt.subplot(3, 3, 7)
    activity_pivot = activity_with_data.set_index('patient_id')['coverage_percentage']
    sleep_pivot = sleep_with_data.set_index('patient_id')['coverage_percentage']
    common_patients = activity_pivot.index.intersection(sleep_pivot.index)
    if len(common_patients) > 10:
        activity_common = activity_pivot.loc[common_patients]
        sleep_common = sleep_pivot.loc[common_patients]
        plt.scatter(activity_common, sleep_common, alpha=0.6, color='purple')
        plt.xlabel('Activity Coverage %')
        plt.ylabel('Sleep Coverage %')
        plt.title('Activity vs Sleep Coverage Correlation', fontweight='bold')
        corr = np.corrcoef(activity_common, sleep_common)[0, 1]
        plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.subplot(3, 3, 8)
    plt.axis('off')
    if not activity_with_data.empty and not sleep_with_data.empty:
        summary_text = f"""
SUMMARY STATISTICS
Activity Data:
• Patients with data: {len(activity_with_data)}/{len(activity_results)} ({len(activity_with_data)/len(activity_results)*100:.1f}%)
• Avg coverage: {activity_with_data['coverage_percentage'].mean():.1f}%
• Avg duration: {activity_with_data['total_duration_hours'].mean():.1f} hours
• Avg time span: {activity_with_data['time_span_hours'].mean():.1f} hours

Sleep Data:
• Patients with data: {len(sleep_with_data)}/{len(sleep_results)} ({len(sleep_with_data)/len(sleep_results)*100:.1f}%)
• Avg coverage: {sleep_with_data['coverage_percentage'].mean():.1f}%
• Avg duration: {sleep_with_data['total_duration_hours'].mean():.1f} hours
• Avg time span: {sleep_with_data['time_span_hours'].mean():.1f} hours

Both Activity and Sleep:
• Patients with both: {len(common_patients)} ({len(common_patients)/len(PATIENT_IDS)*100:.1f}%)
        """
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    #top patients by data quality
    plt.subplot(3, 3, 9)
    plt.axis('off')
    if not activity_with_data.empty and not sleep_with_data.empty:
        if len(common_patients) > 0:
            combined_scores = (activity_pivot.loc[common_patients] + sleep_pivot.loc[common_patients]) / 2
            top_patients = combined_scores.nlargest(10)
            top_text = "TOP 10 PATIENTS BY COMBINED COVERAGE:\n\n"
            for i, (patient_id, score) in enumerate(top_patients.items(), 1):
                act_cov = activity_pivot.loc[patient_id]
                sleep_cov = sleep_pivot.loc[patient_id]
                top_text += f"{i:2d}. Patient {patient_id}: {score:.1f}% (A:{act_cov:.1f}%, S:{sleep_cov:.1f}%)\n"
            plt.text(0.05, 0.95, top_text, transform=plt.gca().transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/continuity_analysis_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    return activity_with_data, sleep_with_data, common_patients

def print_detailed_summary(results_df):
    activity_results = results_df[results_df['data_type'] == 'activity']
    sleep_results = results_df[results_df['data_type'] == 'sleep']
    activity_with_data = activity_results[activity_results['has_data'] == True]
    sleep_with_data = sleep_results[sleep_results['has_data'] == True]
    print("\n" + "="*80)
    print("DETAILED CONTINUITY ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nTOTAL PATIENTS ANALYZED: {len(PATIENT_IDS)}")
    print(f"\nACTIVITY DATA:")
    print(f"  Patients with activity data: {len(activity_with_data)}/{len(activity_results)} ({len(activity_with_data)/len(activity_results)*100:.1f}%)")
    if not activity_with_data.empty:
        print(f"  Coverage percentage: mean={activity_with_data['coverage_percentage'].mean():.1f}%, median={activity_with_data['coverage_percentage'].median():.1f}%")
        print(f"  Total duration (hours): mean={activity_with_data['total_duration_hours'].mean():.1f}, median={activity_with_data['total_duration_hours'].median():.1f}")
        print(f"  Time span (hours): mean={activity_with_data['time_span_hours'].mean():.1f}, median={activity_with_data['time_span_hours'].median():.1f}")
        print(f"  Longest continuous period: mean={activity_with_data['longest_continuous_hours'].mean():.1f}, max={activity_with_data['longest_continuous_hours'].max():.1f} hours")
    print(f"\nSLEEP DATA:")
    print(f"  Patients with sleep data: {len(sleep_with_data)}/{len(sleep_results)} ({len(sleep_with_data)/len(sleep_results)*100:.1f}%)")
    if not sleep_with_data.empty:
        print(f"  Coverage percentage: mean={sleep_with_data['coverage_percentage'].mean():.1f}%, median={sleep_with_data['coverage_percentage'].median():.1f}%")
        print(f"  Total duration (hours): mean={sleep_with_data['total_duration_hours'].mean():.1f}, median={sleep_with_data['total_duration_hours'].median():.1f}")
        print(f"  Time span (hours): mean={sleep_with_data['time_span_hours'].mean():.1f}, median={sleep_with_data['time_span_hours'].median():.1f}")
        print(f"  Longest continuous period: mean={sleep_with_data['longest_continuous_hours'].mean():.1f}, max={sleep_with_data['longest_continuous_hours'].max():.1f} hours")
    high_quality_activity = activity_with_data[activity_with_data['coverage_percentage'] >= 80]
    high_quality_sleep = sleep_with_data[sleep_with_data['coverage_percentage'] >= 80]
    print(f"\nHIGH QUALITY DATA (≥80% coverage):")
    print(f"  Activity: {len(high_quality_activity)} patients ({len(high_quality_activity)/len(activity_results)*100:.1f}%)")
    print(f"  Sleep: {len(high_quality_sleep)} patients ({len(high_quality_sleep)/len(sleep_results)*100:.1f}%)")
    activity_patients = set(activity_with_data['patient_id'])
    sleep_patients = set(sleep_with_data['patient_id'])
    both_datasets = activity_patients.intersection(sleep_patients)
    print(f"\nPATIENTS WITH BOTH DATASETS: {len(both_datasets)} ({len(both_datasets)/len(PATIENT_IDS)*100:.1f}%)")
    return {
        'activity_with_data': len(activity_with_data),
        'sleep_with_data': len(sleep_with_data),
        'both_datasets': len(both_datasets),
        'high_quality_activity': len(high_quality_activity),
        'high_quality_sleep': len(high_quality_sleep)
    }

def main():
    results_df = analyze_all_patients() 
    summary_stats = print_detailed_summary(results_df)
    activity_data, sleep_data, common_patients = create_summary_visualizations(results_df)
    results_df.to_csv('./continuity_analysis/detailed_results.csv', index=False)
    print(f"\nDetailed results saved to: ./continuity_analysis/detailed_results.csv")
    print(f"Visualizations saved to: ./continuity_analysis/continuity_analysis_overview.png")
    return results_df, summary_stats

if __name__ == "__main__":
    results_df, summary_stats = main()