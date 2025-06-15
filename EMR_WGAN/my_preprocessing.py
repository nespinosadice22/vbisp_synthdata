import pandas as pd 
import os 
import numpy as np
def main(): 
    save_folder = "preprocessing"
    os.makedirs(save_folder, exist_ok=True)
    data = pd.read_csv("raw_data/wearable_with_conditions_scalar_from_aligned_no_oxygen_sat.csv")
    data = data.rename(columns={'Type II Diabetes': 'Type 2 Diabetes'})
    data = data.drop(columns=["Unnamed: 0", 
       'Age-related macular degeneration', 'Arthritis', 'Cancer',
       'Cataracts (1+ eyes)', 'Chronic pullmonary problems',
       'Circulation problems', 'Diabetic retinopathy (1+)',
       'Digestive problems', 'Marijuana user', 'Dry eye (1+)', 'Elevated A1C',
       'Glaucoma (1+)', 'Hearing impairment', 'Heart attack',
       'High blood cholesterol', 'High blood pressure', 'Kidney problems',
       'Low blood pressure', 'Mild cognitive impairmen', 'Multiple sclerosis',
       'Obesity', 'Osteoporosis', 'Other heart issues (pacemaker)',
       'Other neurological conditions', "Parkinson's disease", 'Pre-diabetes',
       'Retinal vascular occlusion', 'Stroke',
       'Urinary problems' ])
    print(data.columns)
    print(data.head())

    data = data.dropna()
    #normalize columns 
    min_max_log = {}
    continuous_cols = ['heart_rate_min', 'heart_rate_median',
       'heart_rate_mean', 'heart_rate_max', 'heart_rate_std', 'resp_rate_min',
       'resp_rate_median', 'resp_rate_mean', 'resp_rate_max', 'resp_rate_std',
       'stress_min', 'stress_median', 'stress_mean', 'stress_max',
       'stress_std', 'blood_glucose_min', 'blood_glucose_median',
       'blood_glucose_mean', 'blood_glucose_max', 'blood_glucose_std',
       'act_generic_hrs', 'act_walking_hrs', 'act_running_hrs',
       'act_sedentary_hrs', 'act_generic_total_events',
       'act_walking_total_events', 'act_running_total_events',
       'act_sedentary_total_events', 'total_active_hrs', 'resting_heart_rate',
       'total_steps', 'total_kcal', 'sleep_light_hrs', 'sleep_deep_hrs',
       'sleep_rem_hrs', 'sleep_awake_hrs', 'sleep_light_total_events',
       'sleep_deep_total_events', 'sleep_rem_total_events',
       'sleep_awake_total_events', 'total_sleep_monitor_hrs']
    for col in continuous_cols:
        col_value = np.array(data[col])
        min_max_log[col] = [np.min(col_value), np.max(col_value)]
        norm_col_value = (col_value - min_max_log[col][0]) / (min_max_log[col][1] - min_max_log[col][0])
        data[col] = list(norm_col_value)
    print(data.head())
    print(min_max_log)
    np.save(save_folder+ '/min_max_log.npy', min_max_log)
    
    data2 = data.drop(columns=["patient_id", "day"])
    
    data2.to_csv(save_folder + '/preprocessed_training_data.csv', index=False)
    #split by patient specifically 
    print("Percent of all rows with diabetes:", np.sum(data['Type 2 Diabetes'])/len(data))

   
    np.random.seed(0)
    unique_pids = data["patient_id"].unique()
    np.random.shuffle(unique_pids)
    split_idx = int(len(unique_pids) * 0.7)
    train_pids = unique_pids[:split_idx]
    test_pids = unique_pids[split_idx:] 
    training_data_df = data[data["patient_id"].isin(train_pids)].copy()
    testing_data_df = data[data["patient_id"].isin(test_pids)].copy()
    print("Percent of all training rows with diabetes:", np.sum(training_data_df['Type 2 Diabetes'])/len(training_data_df))
    print("Percent of all testing rows with diabetes:", np.sum(testing_data_df['Type 2 Diabetes'])/len(testing_data_df))
    training_data_df = training_data_df.drop(columns=["patient_id", "day"])
    testing_data_df = testing_data_df.drop(columns=["patient_id", "day"])
    training_data_df.to_csv(save_folder + '/normalized_training_data.csv', index=False)
    testing_data_df.to_csv(save_folder + '/normalized_testing_data.csv', index=False)

    #also save unnormalized version 
    min_max_log = np.load(save_folder + '/min_max_log.npy', allow_pickle=True).item()
    for key, min_max in min_max_log.items():
        min_, max_ = min_max[0], min_max[1]
        col_values = np.array(training_data_df[key])
        training_data_df[key] = (1 - col_values)*min_ + col_values*max_
        col_values = np.array(testing_data_df[key])
        testing_data_df[key] = (1 - col_values)*min_ + col_values*max_
        
    training_data_df.to_csv(save_folder + '/original_training_data.csv', index=False)
    testing_data_df.to_csv(save_folder + '/original_testing_data.csv', index=False)

    df = pd.read_csv("preprocessing/preprocessed_training_data.csv")
    nan_rows = df[df.isnull().any(axis=1)]
    print("Rows with NaN values:")
    print(nan_rows)
    nan_columns = df.columns[df.isnull().any()].tolist()
    print("\nColumns with NaN values:")
    print(nan_columns)


    df = pd.read_csv("preprocessing/normalized_training_data.csv")
    nan_rows = df[df.isnull().any(axis=1)]
    print("Rows with NaN values:")
    print(nan_rows)
    nan_columns = df.columns[df.isnull().any()].tolist()
    print("\nColumns with NaN values:")
    print(nan_columns)

if __name__ == "__main__":
    main()