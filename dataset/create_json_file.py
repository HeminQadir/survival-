
import os
import json
import random
import pandas as pd

def split_list_into_sublists(input_list, k_folds=5):
    # Randomly shuffle the input list
    random.shuffle(input_list)
    
    # Calculate the number of elements per sublist
    elements_per_sublist = len(input_list) // k_folds
    
    # Calculate the remainder to distribute any remaining elements
    remainder = len(input_list) % k_folds
    
    # Initialize the starting index for slicing
    start_index = 0
    
    # Initialize the list to store sublists
    sublists = []
    
    # Iterate through each sublist
    for i in range(k_folds):
        # Calculate the ending index for slicing
        end_index = start_index + elements_per_sublist + (1 if i < remainder else 0)
        
        # Append the sublist to the result list
        sublists.append(input_list[start_index:end_index])
        
        # Update the starting index for the next iteration
        start_index = end_index
    
    return sublists

def create_json(root_directory, file_path, k_folds=5):
    # Get a list of files in "imagesTs" and "labelsTr"
    
    df = pd.read_excel(file_path)
    
    paitents = os.listdir(root_directory)
    #print(paitents)
    
    sublists = split_list_into_sublists(paitents, k_folds=k_folds)

    # Create a list to store the training data
    training_data = []

    # Iterate through each fold
    fold = 0
    for train_index in sublists:
        #train_index = train_index[:4]
        #print(fold)
        #print(train_index)
        #print("*"*40)
        # Select the files for the current fold
        fold_files_images = [os.path.join(root_directory, files_image, "output.nii.gz") for files_image in train_index]
        fold_files_labels = [os.path.join(root_directory, files_image, "seg.nii.gz") for files_image in train_index]

        # Iterate through each file in the current fold
        for file_images, file_labels in zip(fold_files_images, fold_files_labels):
            parts = file_images.split('/')

            patient_id = parts[6]
            #print(patient_id)
            row = df[df['Patient-ID'] == patient_id]

            if row.empty:
                print(f"Warning: No data found for patient {patient_id}")
                continue

            # Retrieve survival time and event status, ensuring no errors
            try:
                survival_time = round(row['overall_survival_months'].values[0], 4)
                event = row['vital_status'].values[0]  # Assuming 'vital_status' is the event indicator
            except IndexError:
                print(f"Warning: Missing survival or event data for patient {patient_id}")
                continue  # Skip this patient


            # Retrieve survival time and event status
            #survival_time =  round(row['overall_survival_months'].values[0], 4)
            #event = row['vital_status'].values[0]  # Assuming 'vital_status' is the event indicator
            example_data = {"fold": fold, "image":  file_images, "label": file_labels, "event": int(event), "time": survival_time}
            training_data.append(example_data)

        fold+=1

    # Create a dictionary with the "training" key and the training data list
    json_data = {"training": training_data}

    # Convert the dictionary to a JSON-formatted string
    json_string = json.dumps(json_data, indent=4)


    # Write the JSON string to a file
    with open("training_data_nikhil_debug.json", "w") as json_file:
        json_file.write(json_string)
            
           
if __name__ == "__main__":
    
    file_path = "/home/nikhil/g17p3/g17-p3/input/crlm.xlsx"  
    #root_directory = "/home/hemin/Survival_Nikhil/data"
    root_directory = "/home/nikhil/g17p3/g17-p3/Converted_NIfTI"  
    create_json(root_directory, file_path)

