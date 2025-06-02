import json, itertools, statistics
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import os 

#----------------------get data frames for a patient by info ------------------------------- # 
def get_bloodglucose(path): 
    path = Path(path)
    with path.open() as f: 
        data = json.load(f) 
    times = [] 
    values = [] 
    for rec in data["body"]["cgm"]: 
        start_time = (rec["effective_time_frame"]["time_interval"]["start_date_time"])
        end_time = (rec["effective_time_frame"]["time_interval"]["end_date_time"])
        if start_time != end_time: 
            print("ERRRORRRR: need to reconcile different start and end times")
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

def get_heartrate(path): 
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

def get_oxygensat(path): 
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

def get_activity(path): 
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

def get_calorie(path): 
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

def get_respiratoryrate(path): 
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

def get_sleep(path): 
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

def get_stress(path): 
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
#-------------------------------------------visualize------------------

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def analyze_temporal_patterns(df, time_col, value_col, sensor_name):
    """
    Analyze and print temporal patterns for a sensor
    """
    if df.empty:
        print(f"\n{sensor_name}: NO DATA")
        return None
    
    # Convert to datetime if not already
    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col])
    df_copy = df_copy.sort_values(time_col)
    
    # Calculate time differences
    time_diffs = df_copy[time_col].diff().dropna()
    
    print(f"\n{sensor_name.upper()} TEMPORAL ANALYSIS:")
    print(f"  Total measurements: {len(df_copy)}")
    print(f"  Time span: {df_copy[time_col].min()} to {df_copy[time_col].max()}")
    print(f"  Duration: {df_copy[time_col].max() - df_copy[time_col].min()}")
    
    if len(time_diffs) > 0:
        intervals_seconds = time_diffs.dt.total_seconds()
        print(f"  Measurement intervals:")
        print(f"    Min: {intervals_seconds.min():.1f} seconds")
        print(f"    Max: {intervals_seconds.max():.1f} seconds") 
        print(f"    Mean: {intervals_seconds.mean():.1f} seconds")
        print(f"    Median: {intervals_seconds.median():.1f} seconds")
        
        # Find most common intervals
        interval_counts = intervals_seconds.value_counts().head(3)
        print(f"  Most common intervals:")
        for interval, count in interval_counts.items():
            print(f"    {interval:.1f}s: {count} times ({count/len(intervals_seconds)*100:.1f}%)")
    
    # Value analysis
    if pd.api.types.is_numeric_dtype(df_copy[value_col]):
        values = df_copy[value_col]
        print(f"  Value statistics:")
        print(f"    Min: {values.min()}")
        print(f"    Max: {values.max()}")
        print(f"    Mean: {values.mean():.2f}")
        print(f"    Std: {values.std():.2f}")
        
        # Check for potential invalid values
        zero_count = (values == 0).sum()
        negative_count = (values < 0).sum()
        if zero_count > 0:
            print(f"    Zero values: {zero_count} ({zero_count/len(values)*100:.1f}%)")
        if negative_count > 0:
            print(f"    Negative values: {negative_count} ({negative_count/len(values)*100:.1f}%)")
    
    return df_copy

def visualize_sensor_data(patient_dataframes, save_dir, patient_id="1023"):
    """
    Create comprehensive visualizations for all sensor data
    """
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    sensor_configs = [
        ('blood_glucose', 'start_time', 'blood_glucose', 'Blood Glucose (mg/dL)', 'tab:red'),
        ('heart_rate', 'date_time', 'heart_rate', 'Heart Rate (BPM)', 'tab:pink'),
        ('oxygen_sat', 'date_time', 'oxygen_saturation', 'Oxygen Saturation (%)', 'tab:blue'),
        ('calorie', 'date_time', 'calories', 'Active Calories (kcal)', 'tab:orange'),
        ('resp_rate', 'date_time', 'respiratory_rate', 'Respiratory Rate (BPM)', 'tab:green'),
        ('stress', 'date_time', 'stress', 'Stress Level', 'tab:purple'),
    ]
    
    # Analyze and plot point-based sensors
    for i, (sensor_key, time_col, value_col, ylabel, color) in enumerate(sensor_configs):
        ax = plt.subplot(4, 2, i+1)
        
        if sensor_key in patient_dataframes and not patient_dataframes[sensor_key].empty:
            df = patient_dataframes[sensor_key].copy()
            
            # Analyze patterns
            df_clean = analyze_temporal_patterns(df, time_col, value_col, sensor_key)
            
            if df_clean is not None:
                # Convert time to datetime
                df_clean[time_col] = pd.to_datetime(df_clean[time_col])
                
                # Plot the data
                plt.scatter(df_clean[time_col], df_clean[value_col], 
                           alpha=0.6, s=10, color=color, label=sensor_key)
                
                # Highlight potentially invalid values
                if sensor_key == 'heart_rate':
                    invalid_mask = df_clean[value_col] == 0
                    if invalid_mask.any():
                        plt.scatter(df_clean[invalid_mask][time_col], 
                                  df_clean[invalid_mask][value_col], 
                                  color='red', s=15, alpha=0.8, label='Invalid (0)')
                
                elif sensor_key in ['resp_rate', 'stress']:
                    invalid_mask = df_clean[value_col] < 0
                    if invalid_mask.any():
                        plt.scatter(df_clean[invalid_mask][time_col], 
                                  df_clean[invalid_mask][value_col], 
                                  color='red', s=15, alpha=0.8, label='Invalid (<0)')
        else:
            plt.text(0.5, 0.5, f'No {sensor_key} data', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=14, color='red')
        
        plt.title(f'{ylabel} - Patient {patient_id}', fontsize=12, fontweight='bold')
        plt.ylabel(ylabel)
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    
    # Handle interval-based data (Activity and Sleep)
    # Activity data
    # Activity data - plotted as scatter points over time
    ax = plt.subplot(4, 2, 7)
    if 'activity' in patient_dataframes and not patient_dataframes['activity'].empty:
        activity_df = patient_dataframes['activity'].copy()
        
        if 'start_time' in activity_df.columns and 'end_time' in activity_df.columns:
            activity_df['start_time'] = pd.to_datetime(activity_df['start_time'])
            activity_df['end_time'] = pd.to_datetime(activity_df['end_time'])
            activity_df['duration'] = (activity_df['end_time'] - activity_df['start_time']).dt.total_seconds() / 60
            activity_df['mid_time'] = activity_df['start_time'] + (activity_df['end_time'] - activity_df['start_time']) / 2
            
            # Get unique activities and assign colors
            unique_activities = activity_df['activity_name'].unique()
            activity_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_activities)))
            color_map = dict(zip(unique_activities, activity_colors))
            
            # Create y-values for different activity types (stacked vertically)
            activity_y_map = {act: i for i, act in enumerate(unique_activities)}
            
            # Plot each activity as scatter points with duration determining size
            for activity_type in unique_activities:
                activity_subset = activity_df[activity_df['activity_name'] == activity_type]
                
                # Use start time for x-axis, activity type index for y-axis
                # Size represents duration (scaled for visibility)
                sizes = activity_subset['duration'] * 2  # Scale factor for visibility
                sizes = np.clip(sizes, 10, 200)  # Limit size range
                
                plt.scatter(activity_subset['start_time'], 
                           [activity_y_map[activity_type]] * len(activity_subset),
                           s=sizes, 
                           c=[color_map[activity_type]], 
                           alpha=0.7, 
                           label=f'{activity_type}',
                           edgecolors='black',
                           linewidth=0.5)
            
            # Create time series showing data availability
            # Plot a continuous line showing when we have ANY activity data
            all_times = []
            for _, row in activity_df.iterrows():
                # Create time points throughout each activity period
                duration_seconds = (row['end_time'] - row['start_time']).total_seconds()
                time_points = pd.date_range(row['start_time'], row['end_time'], 
                                          periods=max(2, int(duration_seconds/300)))  # Point every 5 minutes
                all_times.extend(time_points)
            
            if all_times:
                all_times = sorted(set(all_times))
                # Plot availability line at the top
                y_availability = [len(unique_activities)] * len(all_times)
                plt.plot(all_times, y_availability, 'k-', alpha=0.3, linewidth=2, label='Data Available')
            
            plt.title(f'Activity Data - Patient {patient_id}', fontsize=12, fontweight='bold')
            plt.ylabel('Activity Type')
            plt.xlabel('Time')
            plt.yticks(range(len(unique_activities) + 1), 
                      list(unique_activities) + ['Available'])
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            
        else:
            plt.text(0.5, 0.5, 'No Activity data structure', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=14, color='red')
    else:
        plt.text(0.5, 0.5, 'No Activity data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14, color='red')
    
    # Sleep data - plotted as scatter points over time
    ax = plt.subplot(4, 2, 8)
    if 'sleep' in patient_dataframes and not patient_dataframes['sleep'].empty:
        sleep_df = patient_dataframes['sleep'].copy()
        
        if 'start_time' in sleep_df.columns and 'end_times' in sleep_df.columns:
            sleep_df['start_time'] = pd.to_datetime(sleep_df['start_time'])
            sleep_df['end_times'] = pd.to_datetime(sleep_df['end_times'])
            sleep_df['duration'] = (sleep_df['end_times'] - sleep_df['start_time']).dt.total_seconds() / 60
            sleep_df['mid_time'] = sleep_df['start_time'] + (sleep_df['end_times'] - sleep_df['start_time']) / 2
            
            # Define sleep stage colors and y-positions
            stage_colors = {
                'awake': 'red',
                'light': 'lightblue', 
                'deep': 'darkblue',
                'rem': 'purple'
            }
            
            unique_stages = sleep_df['sleep_stage'].unique()
            stage_y_map = {stage: i for i, stage in enumerate(unique_stages)}
            
            # Plot each sleep stage as scatter points
            for stage in unique_stages:
                stage_subset = sleep_df[sleep_df['sleep_stage'] == stage]
                
                # Size represents duration (scaled for visibility)
                sizes = stage_subset['duration'] * 0.5  # Scale factor for visibility
                sizes = np.clip(sizes, 8, 150)  # Limit size range
                
                color = stage_colors.get(stage, 'gray')
                
                plt.scatter(stage_subset['start_time'], 
                           [stage_y_map[stage]] * len(stage_subset),
                           s=sizes, 
                           c=color, 
                           alpha=0.7, 
                           label=f'{stage.title()}',
                           edgecolors='black',
                           linewidth=0.5)
            
            # Create time series showing sleep data availability
            all_sleep_times = []
            for _, row in sleep_df.iterrows():
                # Create time points throughout each sleep period
                duration_seconds = (row['end_times'] - row['start_time']).total_seconds()
                time_points = pd.date_range(row['start_time'], row['end_times'], 
                                          periods=max(2, int(duration_seconds/300)))  # Point every 5 minutes
                all_sleep_times.extend(time_points)
            
            if all_sleep_times:
                all_sleep_times = sorted(set(all_sleep_times))
                # Plot availability line at the top
                y_availability = [len(unique_stages)] * len(all_sleep_times)
                plt.plot(all_sleep_times, y_availability, 'k-', alpha=0.3, linewidth=2, label='Data Available')
            
            plt.title(f'Sleep Stages - Patient {patient_id}', fontsize=12, fontweight='bold')
            plt.ylabel('Sleep Stage')
            plt.xlabel('Time')
            plt.yticks(range(len(unique_stages) + 1), 
                      [stage.title() for stage in unique_stages] + ['Available'])
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            
        else:
            plt.text(0.5, 0.5, 'No Sleep data structure', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=14, color='red')
    else:
        plt.text(0.5, 0.5, 'No Sleep data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14, color='red')
    
    plt.tight_layout()
    file_name = save_dir + "/sensor_data.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close() 
    
    # Create a summary timeline showing data availability
    create_data_availability_timeline(patient_dataframes, patient_id, save_dir)

def create_data_availability_timeline(patient_dataframes, patient_id, save_dir):
    """
    Create a timeline showing when each sensor has data available
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sensor_info = []
    
    # Point-based sensors
    point_sensors = [
        ('blood_glucose', 'start_time', 'Blood Glucose'),
        ('heart_rate', 'date_time', 'Heart Rate'),
        ('oxygen_sat', 'date_time', 'Oxygen Saturation'),
        ('calorie', 'date_time', 'Calories'),
        ('resp_rate', 'date_time', 'Respiratory Rate'),
        ('stress', 'date_time', 'Stress'),
    ]
    
    for sensor_key, time_col, display_name in point_sensors:
        if sensor_key in patient_dataframes and not patient_dataframes[sensor_key].empty:
            df = patient_dataframes[sensor_key]
            times = pd.to_datetime(df[time_col])
            sensor_info.append({
                'name': display_name,
                'start': times.min(),
                'end': times.max(),
                'count': len(df),
                'type': 'point'
            })
    
    # Interval-based sensors
    if 'activity' in patient_dataframes and not patient_dataframes['activity'].empty:
        activity_df = patient_dataframes['activity']
        if 'start_time' in activity_df.columns and 'end_time' in activity_df.columns:
            start_times = pd.to_datetime(activity_df['start_time'])
            end_times = pd.to_datetime(activity_df['end_time'])
            sensor_info.append({
                'name': 'Activity',
                'start': start_times.min(),
                'end': end_times.max(),
                'count': len(activity_df),
                'type': 'interval'
            })
    
    if 'sleep' in patient_dataframes and not patient_dataframes['sleep'].empty:
        sleep_df = patient_dataframes['sleep']
        if 'start_time' in sleep_df.columns and 'end_times' in sleep_df.columns:
            start_times = pd.to_datetime(sleep_df['start_time'])
            end_times = pd.to_datetime(sleep_df['end_times'])
            sensor_info.append({
                'name': 'Sleep',
                'start': start_times.min(),
                'end': end_times.max(),
                'count': len(sleep_df),
                'type': 'interval'
            })
    
    # Plot the timeline
    if sensor_info:
        # Sort by start time
        sensor_info.sort(key=lambda x: x['start'])
        
        # Find overall time range
        all_starts = [s['start'] for s in sensor_info]
        all_ends = [s['end'] for s in sensor_info]
        overall_start = min(all_starts)
        overall_end = max(all_ends)
        
        print(f"\nDATA AVAILABILITY SUMMARY:")
        print(f"Overall time range: {overall_start} to {overall_end}")
        print(f"Total duration: {overall_end - overall_start}")
        
        # Calculate overlap window (where all sensors have data)
        overlap_start = max(all_starts)
        overlap_end = min(all_ends)
        if overlap_start <= overlap_end:
            print(f"Overlap window: {overlap_start} to {overlap_end}")
            print(f"Overlap duration: {overlap_end - overlap_start}")
        else:
            print("No overlap window - sensors don't have concurrent data")
        
        # Plot each sensor
        colors = plt.cm.Set3(np.linspace(0, 1, len(sensor_info)))
        
        for i, (sensor, color) in enumerate(zip(sensor_info, colors)):
            duration = sensor['end'] - sensor['start']
            
            # Different styles for point vs interval data
            if sensor['type'] == 'point':
                plt.barh(i, duration.total_seconds()/3600, left=sensor['start'], 
                        color=color, alpha=0.7, height=0.6, 
                        label=f"{sensor['name']} ({sensor['count']} points)")
            else:
                plt.barh(i, duration.total_seconds()/3600, left=sensor['start'], 
                        color=color, alpha=0.7, height=0.6, 
                        label=f"{sensor['name']} ({sensor['count']} intervals)",
                        hatch='///')
            
            # Add text with measurement count
            mid_time = sensor['start'] + duration/2
            plt.text(mid_time, i, f"{sensor['count']}", ha='center', va='center', 
                    fontweight='bold', fontsize=10)
        
        # Highlight overlap window if it exists
        if overlap_start <= overlap_end:
            plt.axvspan(overlap_start, overlap_end, alpha=0.2, color='green', 
                       label=f'Overlap Window ({overlap_end - overlap_start})')
        
        plt.yticks(range(len(sensor_info)), [s['name'] for s in sensor_info])
        plt.xlabel('Time')
        plt.title(f'Data Availability Timeline - Patient {patient_id}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        
        plt.tight_layout()
        filename = save_dir + "/data_availability.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close() 
    else:
        print("No sensor data available for timeline")

# Example usage function
def visualize_patient_data(patient_dataframes, save_dir="./plots", patient_id="1023"):
    """
    Main function to call for visualizing patient data
    
    Args:
        patient_dataframes: dict with keys like 'blood_glucose', 'heart_rate', etc.
        patient_id: string identifier for the patient
    """
    print(f"VISUALIZING DATA FOR PATIENT {patient_id}")
    print("="*60)
    
    visualize_sensor_data(patient_dataframes, save_dir, patient_id)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)

def analyze_sensor_intervals_and_gaps(df, time_col, value_col, sensor_name, gap_threshold_minutes=30):
    """
    Detailed analysis of sensor recording intervals and gaps
    """
    if df.empty:
        print(f"\n{sensor_name}: NO DATA")
        return None, None
    
    # Prepare data
    df_clean = df.copy()
    df_clean[time_col] = pd.to_datetime(df_clean[time_col])
    df_clean = df_clean.sort_values(time_col).reset_index(drop=True)
    
    # Calculate intervals between consecutive measurements
    time_diffs = df_clean[time_col].diff().dropna()
    intervals_seconds = time_diffs.dt.total_seconds()
    intervals_minutes = intervals_seconds / 60
    
    print(f"\n{'='*60}")
    print(f"{sensor_name.upper()} - DETAILED INTERVAL ANALYSIS")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"Total measurements: {len(df_clean):,}")
    print(f"Time span: {df_clean[time_col].min()} to {df_clean[time_col].max()}")
    total_duration = df_clean[time_col].max() - df_clean[time_col].min()
    print(f"Total duration: {total_duration}")
    print(f"Average measurement rate: {len(df_clean) / (total_duration.total_seconds() / 3600):.2f} measurements/hour")
    
    # Interval statistics
    print(f"\nINTERVAL STATISTICS:")
    print(f"  Min interval: {intervals_seconds.min():.1f} seconds ({intervals_minutes.min():.2f} minutes)")
    print(f"  Max interval: {intervals_seconds.max():.1f} seconds ({intervals_minutes.max():.2f} minutes)")
    print(f"  Mean interval: {intervals_seconds.mean():.1f} seconds ({intervals_minutes.mean():.2f} minutes)")
    print(f"  Median interval: {intervals_seconds.median():.1f} seconds ({intervals_minutes.median():.2f} minutes)")
    print(f"  Std interval: {intervals_seconds.std():.1f} seconds ({intervals_minutes.std():.2f} minutes)")
    
    # Most common intervals
    print(f"\nMOST COMMON INTERVALS:")
    interval_counts = intervals_seconds.round(0).value_counts().head(10)
    for interval_sec, count in interval_counts.items():
        percentage = (count / len(intervals_seconds)) * 100
        interval_min = interval_sec / 60
        print(f"  {interval_sec:.0f}s ({interval_min:.1f}min): {count:,} times ({percentage:.1f}%)")
    
    # Identify gaps (intervals longer than threshold)
    gap_threshold_seconds = gap_threshold_minutes * 60
    gaps = intervals_seconds[intervals_seconds > gap_threshold_seconds]
    
    print(f"\nGAP ANALYSIS (gaps > {gap_threshold_minutes} minutes):")
    print(f"  Number of gaps: {len(gaps)}")
    if len(gaps) > 0:
        print(f"  Gap duration range: {gaps.min()/60:.1f} to {gaps.max()/60:.1f} minutes")
        print(f"  Mean gap duration: {gaps.mean()/60:.1f} minutes")
        print(f"  Total time in gaps: {gaps.sum()/3600:.2f} hours ({gaps.sum()/total_duration.total_seconds()*100:.1f}% of total time)")
        
        # Show largest gaps
        gap_indices = intervals_seconds[intervals_seconds > gap_threshold_seconds].index
        largest_gaps = []
        for idx in gap_indices:
            gap_start = df_clean.iloc[idx-1][time_col]
            gap_end = df_clean.iloc[idx][time_col]
            gap_duration = (gap_end - gap_start).total_seconds() / 60
            largest_gaps.append({
                'start': gap_start,
                'end': gap_end,
                'duration_minutes': gap_duration
            })
        
        # Sort by duration and show top 5
        largest_gaps.sort(key=lambda x: x['duration_minutes'], reverse=True)
        print(f"\n  LARGEST GAPS:")
        for i, gap in enumerate(largest_gaps[:5]):
            print(f"    {i+1}. {gap['start']} to {gap['end']} ({gap['duration_minutes']:.1f} minutes)")
    
    # Recording pattern analysis
    print(f"\nRECORDING PATTERN ANALYSIS:")
    
    # Group by hour of day
    df_clean['hour'] = df_clean[time_col].dt.hour
    hourly_counts = df_clean['hour'].value_counts().sort_index()
    most_active_hours = hourly_counts.nlargest(3)
    least_active_hours = hourly_counts.nsmallest(3)
    
    print(f"  Most active hours: {', '.join([f'{h}:00 ({c} measurements)' for h, c in most_active_hours.items()])}")
    print(f"  Least active hours: {', '.join([f'{h}:00 ({c} measurements)' for h, c in least_active_hours.items()])}")
    
    # Group by day
    df_clean['date'] = df_clean[time_col].dt.date
    daily_counts = df_clean['date'].value_counts().sort_index()
    
    print(f"  Recording days: {len(daily_counts)} unique days")
    print(f"  Measurements per day: min={daily_counts.min()}, max={daily_counts.max()}, mean={daily_counts.mean():.1f}")
    
    # Days with no or very few measurements
    low_activity_threshold = daily_counts.mean() * 0.1  # Less than 10% of average
    low_activity_days = daily_counts[daily_counts < low_activity_threshold]
    if len(low_activity_days) > 0:
        print(f"  Low activity days ({len(low_activity_days)} days with <{low_activity_threshold:.0f} measurements):")
        for date, count in low_activity_days.items():
            print(f"    {date}: {count} measurements")
    
    return {
        'intervals_seconds': intervals_seconds,
        'gaps': gaps,
        'gap_indices': gap_indices if len(gaps) > 0 else [],
        'hourly_pattern': hourly_counts,
        'daily_pattern': daily_counts,
        'largest_gaps': largest_gaps if len(gaps) > 0 else []
    }, df_clean

def create_individual_sensor_plots(sensor_name, df_clean, analysis, patient_id, save_dir):
    """
    Create 3 separate plots for a single sensor and save them
    """
    time_col = 'start_time' if 'start_time' in df_clean.columns else 'date_time'
    value_col = [col for col in df_clean.columns if col not in [time_col, 'hour', 'date']][0]
    
    # Plot 1: Time series with gaps highlighted
    plt.figure(figsize=(12, 6))
    plt.scatter(df_clean[time_col], df_clean[value_col], alpha=0.6, s=12, color='blue')
    
    # Highlight gaps
    if len(analysis['gap_indices']) > 0:
        for gap_idx in analysis['gap_indices']:
            gap_start = df_clean.iloc[gap_idx-1][time_col]
            gap_end = df_clean.iloc[gap_idx][time_col]
            plt.axvspan(gap_start, gap_end, alpha=0.3, color='red', label='Gap' if gap_idx == analysis['gap_indices'][0] else "")
    
    plt.title(f'Patient {patient_id} - {sensor_name.replace("_", " ").title()} Time Series', fontsize=14, fontweight='bold')
    plt.ylabel(value_col.replace('_', ' ').title())
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    total_duration_hours = (df_clean[time_col].max() - df_clean[time_col].min()).total_seconds() / 3600
    if total_duration_hours <= 24:
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(total_duration_hours/8))))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    elif total_duration_hours <= 168:
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=max(6, int(total_duration_hours/12))))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    else:
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(total_duration_hours/24/8))))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    plt.xticks(rotation=45)
    if len(analysis['gap_indices']) > 0:
        plt.legend()
    
    plt.tight_layout()
    filename1 = f"{save_dir}/patient_{patient_id}_{sensor_name}_timeseries.png"
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename1}")
    plt.close()
    
    # Plot 2: Interval histogram
    plt.figure(figsize=(10, 6))
    intervals_minutes = analysis['intervals_seconds'] / 60
    
    max_interval = intervals_minutes.max()
    if max_interval < 60:
        bins = np.linspace(0, max_interval, 50)
    else:
        bins = np.logspace(np.log10(max(intervals_minutes.min(), 0.1)), np.log10(max_interval), 50)
        plt.xscale('log')
    
    plt.hist(intervals_minutes, bins=bins, alpha=0.7, color='green', edgecolor='black')
    plt.title(f'Patient {patient_id} - {sensor_name.replace("_", " ").title()} Interval Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Interval (minutes)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for common intervals
    median_interval = intervals_minutes.median()
    mean_interval = intervals_minutes.mean()
    plt.axvline(median_interval, color='red', linestyle='--', alpha=0.8, label=f'Median: {median_interval:.1f}min')
    plt.axvline(mean_interval, color='orange', linestyle='--', alpha=0.8, label=f'Mean: {mean_interval:.1f}min')
    plt.legend()
    
    plt.tight_layout()
    filename2 = f"{save_dir}/patient_{patient_id}_{sensor_name}_intervals.png"
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename2}")
    plt.close()
    
    # Plot 3: Hourly pattern
    plt.figure(figsize=(10, 6))
    
    hours = range(24)
    hourly_counts = [analysis['hourly_pattern'].get(h, 0) for h in hours]
    
    bars = plt.bar(hours, hourly_counts, alpha=0.7, color='purple')
    plt.title(f'Patient {patient_id} - {sensor_name.replace("_", " ").title()} Hourly Recording Pattern', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Measurements')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)
    
    # Highlight peak hours
    max_count = max(hourly_counts) if hourly_counts else 0
    if max_count > 0:
        for j, (hour, count) in enumerate(zip(hours, hourly_counts)):
            if count > max_count * 0.8:
                bars[j].set_color('red')
                bars[j].set_alpha(0.9)
    
    plt.tight_layout()
    filename3 = f"{save_dir}/patient_{patient_id}_{sensor_name}_hourly.png"
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename3}")
    plt.close()

def create_gap_timeline(sensor_data, sensor_analyses, patient_id, save_dir):
    """
    Create timeline showing gaps across all sensors and save it
    """
    plt.figure(figsize=(16, 8))
    
    valid_sensors = [(name, sensor_data[name], sensor_analyses[name]) for name in sensor_data.keys() 
                     if sensor_analyses[name] is not None]
    
    if not valid_sensors:
        print("No valid sensor data for gap timeline")
        return
    
    # Find overall time range
    all_starts = []
    all_ends = []
    for sensor_name, df_clean, analysis in valid_sensors:
        time_col = 'start_time' if 'start_time' in df_clean.columns else 'date_time'
        all_starts.append(df_clean[time_col].min())
        all_ends.append(df_clean[time_col].max())
    
    overall_start = min(all_starts)
    overall_end = max(all_ends)
    
    print(f"Timeline: {overall_start} to {overall_end}")
    
    # Plot each sensor's data availability and gaps
    colors = plt.cm.Set3(np.linspace(0, 1, len(valid_sensors)))
    
    for i, ((sensor_name, df_clean, analysis), color) in enumerate(zip(valid_sensors, colors)):
        time_col = 'start_time' if 'start_time' in df_clean.columns else 'date_time'
        
        # Plot data availability
        sensor_start = df_clean[time_col].min()
        sensor_end = df_clean[time_col].max()
        duration_hours = (sensor_end - sensor_start).total_seconds() / 3600
        
        plt.barh(i, duration_hours, left=sensor_start, color=color, alpha=0.6, height=0.4, 
                label=f'{sensor_name.replace("_", " ").title()} ({len(df_clean):,} points)')
        
        # Plot gaps as red bars
        if len(analysis['largest_gaps']) > 0:
            for gap in analysis['largest_gaps']:
                gap_duration_hours = gap['duration_minutes'] / 60
                plt.barh(i, gap_duration_hours, left=gap['start'], color='red', alpha=0.8, height=0.6)
                
                # Add text for large gaps
                if gap['duration_minutes'] > 120:
                    gap_mid = gap['start'] + (gap['end'] - gap['start']) / 2
                    plt.text(gap_mid, i, f"{gap['duration_minutes']:.0f}m", ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='white')
    
    # Add timeline markers
    plt.axvline(overall_start, color='green', linestyle='--', alpha=0.7, label=f'First: {overall_start.strftime("%m/%d %H:%M")}')
    plt.axvline(overall_end, color='orange', linestyle='--', alpha=0.7, label=f'Last: {overall_end.strftime("%m/%d %H:%M")}')
    
    # Formatting
    plt.yticks(range(len(valid_sensors)), [name.replace('_', ' ').title() for name, _, _ in valid_sensors])
    plt.xlabel('Time')
    plt.title(f'Data Availability and Gaps Timeline - Patient {patient_id}\n(Red bars = gaps, numbers = gap duration in minutes)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis
    total_duration = (overall_end - overall_start).total_seconds() / 3600
    if total_duration <= 24:
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(total_duration/8))))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    elif total_duration <= 168:
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=max(6, int(total_duration/12))))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    else:
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(total_duration/24/8))))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    filename = f"{save_dir}/patient_{patient_id}_gap_timeline.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved gap timeline: {filename}")
    plt.close()

def analyze_patient_intervals_and_gaps(patient_dataframes, patient_id="1023", gap_threshold_minutes=30, save_dir="./plots"):
    """
    Main function to perform detailed interval and gap analysis with separate saved plots
    """
    print(f"DETAILED INTERVAL AND GAP ANALYSIS FOR PATIENT {patient_id}")
    print(f"Gap threshold: {gap_threshold_minutes} minutes")
    print(f"Saving plots to: {save_dir}")
    print("="*80)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Define sensor configurations
    sensor_configs = [
        ('blood_glucose', 'start_time', 'blood_glucose'),
        ('heart_rate', 'date_time', 'heart_rate'),
        ('oxygen_sat', 'date_time', 'oxygen_saturation'),
        ('calorie', 'date_time', 'calories'),
        ('resp_rate', 'date_time', 'respiratory_rate'),
        ('stress', 'date_time', 'stress'),
    ]
    
    sensor_analyses = {}
    sensor_data = {}
    
    # Analyze each sensor
    for sensor_key, time_col, value_col in sensor_configs:
        if sensor_key in patient_dataframes and not patient_dataframes[sensor_key].empty:
            analysis, df_clean = analyze_sensor_intervals_and_gaps(
                patient_dataframes[sensor_key], 
                time_col, 
                value_col, 
                sensor_key,
                gap_threshold_minutes
            )
            sensor_analyses[sensor_key] = analysis
            sensor_data[sensor_key] = df_clean
        else:
            sensor_analyses[sensor_key] = None
            sensor_data[sensor_key] = None
    
    # Handle interval-based sensors (activity, sleep)
    for sensor_key in ['activity', 'sleep']:
        if sensor_key in patient_dataframes and not patient_dataframes[sensor_key].empty:
            df = patient_dataframes[sensor_key]
            print(f"\n{'='*60}")
            print(f"{sensor_key.upper()} - INTERVAL-BASED DATA ANALYSIS")
            print(f"{'='*60}")
            
            if sensor_key == 'activity' and 'start_time' in df.columns and 'end_time' in df.columns:
                df['start_time'] = pd.to_datetime(df['start_time'])
                df['end_time'] = pd.to_datetime(df['end_time'])
                df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
                
                print(f"Total activity intervals: {len(df)}")
                print(f"Time span: {df['start_time'].min()} to {df['end_time'].max()}")
                print(f"Activity types: {df['activity_name'].value_counts().to_dict()}")
                print(f"Interval durations (minutes): min={df['duration_minutes'].min():.1f}, max={df['duration_minutes'].max():.1f}, mean={df['duration_minutes'].mean():.1f}")
            
            elif sensor_key == 'sleep' and 'start_time' in df.columns and 'end_times' in df.columns:
                df['start_time'] = pd.to_datetime(df['start_time'])
                df['end_times'] = pd.to_datetime(df['end_times'])
                df['duration_minutes'] = (df['end_times'] - df['start_time']).dt.total_seconds() / 60
                
                print(f"Total sleep intervals: {len(df)}")
                print(f"Time span: {df['start_time'].min()} to {df['end_times'].max()}")
                print(f"Sleep stages: {df['sleep_stage'].value_counts().to_dict()}")
                print(f"Interval durations (minutes): min={df['duration_minutes'].min():.1f}, max={df['duration_minutes'].max():.1f}, mean={df['duration_minutes'].mean():.1f}")
    
    # Create and save visualizations
    print(f"\n{'='*80}")
    print("CREATING AND SAVING INDIVIDUAL PLOTS...")
    print(f"{'='*80}")
    
    valid_sensors = [(name, sensor_data[name], sensor_analyses[name]) for name in sensor_data.keys() 
                     if sensor_analyses[name] is not None]
    
    for sensor_name, df_clean, analysis in valid_sensors:
        print(f"\nCreating plots for {sensor_name}...")
        create_individual_sensor_plots(sensor_name, df_clean, analysis, patient_id, save_dir)
    
    # Create gap timeline
    print(f"\nCreating gap timeline...")
    create_gap_timeline(sensor_data, sensor_analyses, patient_id, save_dir)
    
    # Summary recommendations
    print(f"\n{'='*80}")
    print("ALIGNMENT RECOMMENDATIONS:")
    print(f"{'='*80}")
    
    valid_analyses = {k: v for k, v in sensor_analyses.items() if v is not None}
    
    if valid_analyses:
        all_intervals = []
        for sensor_name, analysis in valid_analyses.items():
            median_interval = analysis['intervals_seconds'].median()
            all_intervals.append((sensor_name, median_interval))
            print(f"{sensor_name}: median interval = {median_interval:.0f} seconds ({median_interval/60:.1f} minutes)")
        
        intervals_only = [interval for _, interval in all_intervals]
        finest_interval = min(intervals_only)
        coarsest_interval = max(intervals_only)
        
        print(f"\nSuggested alignment strategies:")
        print(f"  Finest resolution: {finest_interval:.0f} seconds ({finest_interval/60:.1f} minutes)")
        print(f"  Balanced resolution: {np.median(intervals_only):.0f} seconds ({np.median(intervals_only)/60:.1f} minutes)")
        print(f"  Coarsest resolution: {coarsest_interval:.0f} seconds ({coarsest_interval/60:.1f} minutes)")
    
    print(f"\n{'='*80}")
    print(f"ALL PLOTS SAVED TO: {save_dir}/")
    print(f"{'='*80}")
    
    return sensor_analyses, sensor_data

def get_patient_data(patient): 
    root = Path("~/Desktop/synth_data/dataset").expanduser()
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
    blood_glucose = get_bloodglucose(files["blood_glucose"])
    heart_rate = get_heartrate(files["heart_rate"])
    oxygen_sat = get_oxygensat(files["oxygen_sat"])
    activity = get_activity(files["activity"])
    calorie = get_calorie(files["calorie"])
    resp_rate = get_respiratoryrate(files["resp_rate"])
    sleep = get_sleep(files["sleep"])
    stress = get_stress(files["stress"])
    patient_dataframes = {
        'blood_glucose': blood_glucose,
        'heart_rate': heart_rate,
        'oxygen_sat': oxygen_sat,
        'activity': activity,
        'calorie': calorie,
        'resp_rate': resp_rate,
        'sleep': sleep,
        'stress': stress
    }
    return patient_dataframes 


def main(): 
    patient = "1027"
    patient_data = get_patient_data(patient)
    print(patient_data['blood_glucose'].head(5))
    print(patient_data['heart_rate'].head(5))
    print(patient_data['oxygen_sat'].head(5))
    print(patient_data['activity'].head(5))
    print(patient_data['calorie'].head(5))
    os.makedirs(f"./patient_{patient}_plots", exist_ok = True)
    visualize_patient_data(patient_data, save_dir=f"./patient_{patient}_plots", patient_id=patient)

    analyses, cleaned_data = analyze_patient_intervals_and_gaps(
        patient_data, 
        patient_id=patient,
        gap_threshold_minutes=30,
        save_dir=f"./patient_{patient}_plots"
    )

    


if __name__ == "__main__":
    main() 