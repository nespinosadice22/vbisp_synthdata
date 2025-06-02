
import os
import pandas as pd 

def get_patients(directory_path, participant_ids): 
    if os.path.exists(directory_path):
        folders_in_directory = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
        try:
            folder_ids = [int(folder) for folder in folders_in_directory if folder.isdigit()]
        except ValueError:
            folder_ids = folders_in_directory
    else:
        print(f"Directory {directory_path} does not exist!")
        folder_ids = []

    participant_ids_set = set(participant_ids)
    folder_ids_set = set(folder_ids)

    #get missing and extra
    missing_in_directory = participant_ids_set - folder_ids_set
    extra_in_directory = folder_ids_set - participant_ids_set

    print(f"Total participant IDs in CSV: {len(participant_ids)}")
    print(f"Total folders in directory: {len(folder_ids)}")
    print(f"Participant IDs missing from directory: {len(missing_in_directory)}")
    print(f"Extra folders in directory (not in CSV): {len(extra_in_directory)}")

    if missing_in_directory:
        print(f"Missing participant IDs: {sorted(missing_in_directory)}")
    if extra_in_directory:
        print(f"Extra folders: {sorted(extra_in_directory)}")
    all_present = len(missing_in_directory) == 0
    print(f"All participant IDs present in directory: {all_present}")
    return missing_in_directory, extra_in_directory
    
def main(): 
    #file paths 
    blood_glucose = "../dataset/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6"
    heart_rate = "../dataset/wearable_activity_monitor/heart_rate/garmin_vivosmart5"
    oxygen_sat = "../dataset/wearable_activity_monitor/oxygen_saturation/garmin_vivosmart5"
    activity = "../dataset/wearable_activity_monitor/physical_activity/garmin_vivosmart5"
    calorie = "../dataset/wearable_activity_monitor/physical_activity_calorie/garmin_vivosmart5"
    respiratory_rate = "../dataset/wearable_activity_monitor/respiratory_rate/garmin_vivosmart5"
    sleep = "../dataset/wearable_activity_monitor/sleep/garmin_vivosmart5"
    stress = "../dataset/wearable_activity_monitor/stress/garmin_vivosmart5"

   
    #check for missing patients 
    patients = pd.read_csv("manifest.csv")
    participant_ids = patients['participant_id'].tolist() 

    print("HEART RATE: ")
    hr_missing, hr_extra = get_patients(heart_rate, participant_ids)
    print("OXYGEN SATS: ")
    oxygensat_missing, oxygensat_extra = get_patients(oxygen_sat, participant_ids)
    print("PHYSICAL ACTIVITY: ")
    act_missing, act_extra = get_patients(activity, participant_ids)
    print("CALORIE: ")
    cal_missing, cal_extra  = get_patients(calorie, participant_ids)
    print("RESPIRATORY RATE: ")
    rr_missing, rr_extra = get_patients(respiratory_rate, participant_ids)
    print("SLEEP: ")
    sleep_missing, sleep_extra = get_patients(sleep, participant_ids)
    print("STRESS: ")
    stress_missing, stress_extra = get_patients(stress, participant_ids)
    print("BLOOD GLUCOSE: ")
    blood_glucose_missing, blood_glucose_extra = get_patients(blood_glucose, participant_ids)

    #FIND THOSE WHO WE HAVE EVERYTHING FOR 
    all_participants = set(participant_ids)
    participants_with_everything = (all_participants - hr_missing - oxygensat_missing - act_missing - cal_missing - rr_missing - sleep_missing - stress_missing - blood_glucose_missing)
    print(f"\nPARTICIPANTS WITH EVERYTHING (all data types): {len(participants_with_everything)}")
    print(f"IDs: {sorted(participants_with_everything)}")
    
    # Find participants who have everything EXCEPT oxygen sats
    participants_everything_but_oxygen = (all_participants - hr_missing - act_missing - cal_missing - rr_missing - sleep_missing - stress_missing - blood_glucose_missing)
    print(f"\nPARTICIPANTS WITH EVERYTHING BUT OXYGEN SATS: {len(participants_everything_but_oxygen)}")
    print(f"IDs: {sorted(participants_everything_but_oxygen)}")


    
  
    

if __name__ == "__main__": 
    main() 