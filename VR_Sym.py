# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:29:54 2024

@author: tsiragy
"""

import pandas as pd
import matplotlib as mtlp
import numpy as np
import scipy as sc
import os
import statistics 

import pymatreader
from pymatreader import read_mat

import scipy.stats as stats
from statsmodels.stats.anova import AnovaRM


import seaborn as sns
import matplotlib.pyplot as plt


def read_data(directory):
    # Dictionary to store the DataFrame from each file
    dataframes = {}

    # List all files in the directory
    for filename in os.listdir(directory):
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)

        # Ensure we only read text files
        if os.path.isfile(filepath) and filename.endswith('.mat'):
            # Read the content into a DataFrame using comma as the delimiter
            df = read_mat(filepath)

            # Store the DataFrame in the dictionary
            dataframes[filename] = df

    return dataframes

directory = 'C:/Users/tsira/Documents/OneDrive_2024-07-22/Old_Data'

dataframes = read_data(directory)



# The suffix you're looking for
suffix = '_firstMinute.mat'
suffix_2 = '_lastMinute.mat'

# Extract keys with the specific suffix
keys_with_suffix = [key for key in dataframes if key.endswith(suffix)]
keys_with_suffix_2 = [key for key in dataframes if key.endswith(suffix_2)]

# Extract keys and their associated values with the specific suffix
first = {key: value for key, value in dataframes.items() if key.endswith(suffix)}
last = {key: value for key, value in dataframes.items() if key.endswith(suffix_2)}

Left_ST = 'L_StrideTime'
Left_ST_std = 'L_StrideTime_std'
Left_SL = 'L_StrideLength'
Left_SL_std = 'L_StrideLength_std'
Left_SW = 'L_StepWidth'
Left_SW_std = 'L_StepWidth_std'



Right_ST = 'R_StrideTime'
Right_ST_std = 'R_StrideTime_std'
Right_SL = 'R_StrideLength'
Right_SL_std = 'R_StrideLength_std'
Right_SW = 'R_StepWidth'
Right_SW_std = 'R_StepWidth_std'



# Extract the specific parameter from each nested dictionary
Left_StrideT_first = {k: v[Left_ST] for k, v in first.items()}
Left_StrideL_first = {k: v[Left_SL] for k, v in first.items()}
Left_StepW_first = {k: v[Left_SW] for k, v in first.items()}
    

# Extract the specific parameter from each nested dictionary
Right_StrideT_first = {k: v[Right_ST] for k, v in first.items()}
Right_StrideL_first = {k: v[Right_SL] for k, v in first.items()}
Right_StepW_first = {k: v[Right_SW] for k, v in first.items()}
    


# Extract the specific parameter from each nested dictionary
Left_StrideT_last = {k: v[Left_ST] for k, v in last.items()}
Left_StrideL_last = {k: v[Left_SL] for k, v in last.items()}
Left_StepW_last = {k: v[Left_SW] for k, v in last.items()}
    

# Extract the specific parameter from each nested dictionary
Right_StrideT_last = {k: v[Right_ST] for k, v in last.items()}
Right_StrideL_last = {k: v[Right_SL] for k, v in last.items()}
Right_StepW_last = {k: v[Right_SW] for k, v in last.items()}
    





# Extract the specific parameter from each nested dictionary
Left_StrideT_first_SD = {k: v[Left_ST_std] for k, v in first.items()}
Left_StrideL_first_SD = {k: v[Left_SL_std] for k, v in first.items()}
Left_StepW_first_SD = {k: v[Left_SW_std] for k, v in first.items()}
    

# Extract the specific parameter from each nested dictionary
Right_StrideT_first_SD = {k: v[Right_ST_std] for k, v in first.items()}
Right_StrideL_first_SD = {k: v[Right_SL_std] for k, v in first.items()}
Right_StepW_first_SD = {k: v[Right_SW_std] for k, v in first.items()}
    


# Extract the specific parameter from each nested dictionary
Left_StrideT_last_SD = {k: v[Left_ST_std] for k, v in last.items()}
Left_StrideL_last_SD = {k: v[Left_SL_std] for k, v in last.items()}
Left_StepW_last_SD = {k: v[Left_SW_std] for k, v in last.items()}
    

# Extract the specific parameter from each nested dictionary
Right_StrideT_last_SD = {k: v[Right_ST_std] for k, v in last.items()}
Right_StrideL_last_SD = {k: v[Right_SL_std] for k, v in last.items()}
Right_StepW_last_SD = {k: v[Right_SW_std] for k, v in last.items()}
    











# Convert left values from dictionary to numpy array
values = Left_StrideT_first.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrT_first = np.array(values_list)


values = Left_StrideL_first.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrL_first  = np.array(values_list)


values = Left_StepW_first.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StepW_first  = np.array(values_list)




values = Left_StrideT_last.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrT_last = np.array(values_list)


values = Left_StrideL_last.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrL_last  = np.array(values_list)


values = Left_StepW_last.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StepW_last  = np.array(values_list)







# Convert right values from dictionary to numpy array
values = Right_StrideT_first.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrT_first = np.array(values_list)


values = Right_StrideL_first.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrL_first  = np.array(values_list)


values = Right_StepW_first.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StepW_first  = np.array(values_list)


values = Right_StrideT_last.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrT_last = np.array(values_list)


values = Right_StrideL_last.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrL_last  = np.array(values_list)


values = Right_StepW_last.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StepW_last  = np.array(values_list)






# Convert left values from dictionary to numpy array
values = Left_StrideT_first_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrT_first_SD = np.array(values_list)


values = Left_StrideL_first_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrL_first_SD  = np.array(values_list)


values = Left_StepW_first_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StepW_first_SD  = np.array(values_list)




values = Left_StrideT_last_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrT_last_SD = np.array(values_list)


values = Left_StrideL_last_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrL_last_SD  = np.array(values_list)


values = Left_StepW_last_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StepW_last_SD  = np.array(values_list)







# Convert right values from dictionary to numpy array
values = Right_StrideT_first_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrT_first_SD = np.array(values_list)


values = Right_StrideL_first_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrL_first_SD  = np.array(values_list)


values = Right_StepW_first_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StepW_first_SD  = np.array(values_list)


values = Right_StrideT_last_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrT_last_SD = np.array(values_list)


values = Right_StrideL_last_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrL_last_SD  = np.array(values_list)


values = Right_StepW_last_SD.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StepW_last_SD  = np.array(values_list)





# Import Baseline Data 
B_directory = 'C:/Users/tsira/Documents/OneDrive_2024-07-22/Old_Data/Baseline_Data'
B_dataframes = read_data(B_directory)



# Extract the specific parameter from each nested dictionary
Left_StrideT_B = {k: v[Left_ST] for k, v in B_dataframes.items()}
Left_StrideL_B = {k: v[Left_SL] for k, v in B_dataframes.items()}
Left_StepW_B = {k: v[Left_SW] for k, v in B_dataframes.items()}
    

# Extract the specific parameter from each nested dictionary
Right_StrideT_B = {k: v[Right_ST] for k, v in B_dataframes.items()}
Right_StrideL_B = {k: v[Right_SL] for k, v in B_dataframes.items()}
Right_StepW_B = {k: v[Right_SW] for k, v in B_dataframes.items()}
    




# Extract the specific parameter from each nested dictionary
Left_StrideT_SD_B = {k: v[Left_ST_std] for k, v in B_dataframes.items()}
Left_StrideL_SD_B = {k: v[Left_SL_std] for k, v in B_dataframes.items()}
Left_StepW_SD_B = {k: v[Left_SW_std] for k, v in B_dataframes.items()}
    

# Extract the specific parameter from each nested dictionary
Right_StrideT_SD_B = {k: v[Right_ST_std] for k, v in B_dataframes.items()}
Right_StrideL_SD_B = {k: v[Right_SL_std] for k, v in B_dataframes.items()}
Right_StepW_SD_B = {k: v[Right_SW_std] for k, v in B_dataframes.items()}
    








values = Left_StrideT_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrT_B = np.array(values_list)


values = Left_StrideL_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrL_B  = np.array(values_list)


values = Left_StepW_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StepW_B  = np.array(values_list)



values = Right_StrideT_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrT_B = np.array(values_list)


values = Right_StrideL_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrL_B  = np.array(values_list)


values = Right_StepW_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StepW_B  = np.array(values_list)














values = Left_StrideT_SD_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrT_SD_B = np.array(values_list)


values = Left_StrideL_SD_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StrL_SD_B  = np.array(values_list)


values = Left_StepW_SD_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StepW_SD_B  = np.array(values_list)



values = Right_StrideT_SD_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrT_SD_B = np.array(values_list)


values = Right_StrideL_SD_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StrL_SD_B  = np.array(values_list)


values = Right_StepW_SD_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StepW_SD_B  = np.array(values_list)














# Calculate Gait Asymmetry for first and last minute of VR

StrT_first = np.absolute(np.array(Lft_StrT_first) - np.array(Rgt_StrT_first))
StrL_first = np.absolute(np.array(Lft_StrL_first) - np.array(Rgt_StrL_first))
StpW_first = np.absolute(np.array(Lft_StepW_first) - np.array(Rgt_StepW_first))

StrT_last = np.absolute(np.array(Lft_StrT_last) - np.array(Rgt_StrT_last))
StrL_last = np.absolute(np.array(Lft_StrL_last) - np.array(Rgt_StrL_last))
StpW_last = np.absolute(np.array(Lft_StepW_last) - np.array(Rgt_StepW_last))

# Calcualte Gait Asymmetry for baseline walking


StrT_B = np.absolute(np.array(Lft_StrT_B) - np.array(Rgt_StrT_B))
StrL_B = np.absolute(np.array(Lft_StrL_B) - np.array(Rgt_StrL_B))
StpW_B = np.absolute(np.array(Lft_StepW_B) - np.array(Rgt_StepW_B))







# reshape the data
StrT_B = StrL_B .reshape(-1, 1)  # Shape will be (16, 1)
StrL_B = StrL_B.reshape(-1,1)
StpW_B = StpW_B.reshape(-1,1)

StrT_first = StrT_first.reshape(-1,1)
StrL_first = StrL_first.reshape(-1,1)
StpW_first = StpW_first.reshape(-1,1)

StrT_last = StrT_last.reshape(-1,1)
StrL_last = StrL_last.reshape(-1,1)
StpW_last = StpW_last.reshape(-1,1)







#check for statistical outliers and normality
data_length = np.hstack((StrL_B, StrL_first, StrL_last))






def remove_outliers(arr):
    """
    Remove outliers from a 2D NumPy array, column-wise, where outliers are defined as
    values that are 3 standard deviations above or below the mean.
    Returns both the filtered array and the indices of removed outliers.

    Parameters:
    arr (np.ndarray): 2D NumPy array where each column is treated independently.

    Returns:
    np.ndarray: A new NumPy array with outliers removed.
    list: A list of indices of the removed outliers.
    """
    # Copy the array to avoid modifying the original one
    filtered_arr = arr.copy()
    
    # Initialize a mask that will be True for all rows
    mask = np.ones(arr.shape[0], dtype=bool)
    
    # Initialize a set to keep track of outlier indices
    removed_indices = set()
    
    # Iterate over each column
    for i in range(arr.shape[1]):
        # Compute mean and standard deviation for the column
        column = arr[:, i]
        mean = np.mean(column)
        std_dev = np.std(column)
        
        # Define the boundaries for outliers
        lower_bound = mean - 3 * std_dev  #should be 3, if 4 is there then this is purely for testing
        upper_bound = mean + 3 * std_dev
        
        # Update the mask to filter out outliers in the current column
        column_mask = (column >= lower_bound) & (column <= upper_bound)
        
        # Find the indices of outliers in this column
        column_outlier_indices = np.where(~column_mask)[0]
        removed_indices.update(column_outlier_indices)
        
        # Combine masks for all columns
        mask &= column_mask
    
    # Apply mask to filter out outliers
    filtered_arr = filtered_arr[mask]
    
    # Convert the set of removed indices to a list
    removed_indices = list(removed_indices)
    
    return filtered_arr, removed_indices

    

def filter_other_arrays(arrays, removed_indices):
    """
    Filter out rows from multiple arrays based on the indices of removed outliers from the primary array.

    Parameters:
    arrays (list of np.ndarray): List of 2D NumPy arrays to filter.
    removed_indices (list): List of indices of removed outliers.

    Returns:
    list of np.ndarray: List of filtered arrays.
    """
    # Create a mask for remaining indices (invert the removed_indices)
    all_indices = np.arange(arrays[0].shape[0])
    remaining_indices = np.setdiff1d(all_indices, removed_indices)

    # Apply the mask to all arrays
    filtered_arrays = [arr[remaining_indices] for arr in arrays]
    
    return filtered_arrays
    
    
    
# Remove outliers and get indices of removed outliers
filtered_Len, removed_indices = remove_outliers(StrL_B)    

# Filter other arrays based on removed indices for Stride Length
filtered_Len2, filtered_Len3 = filter_other_arrays([StrL_first, StrL_last], removed_indices)
    

# Filter other arrays based on removed indices for Stride Time
filtered_Time, filtered_Time2, filtered_Time3 = filter_other_arrays([StrT_B,StrT_first, StrT_last], removed_indices)

# Filter other arrays based on removed indices for Step Width
filtered_Width, filtered_Width2, filtered_Width3 = filter_other_arrays([StpW_B,StpW_first, StpW_last], removed_indices)

    
    
    
    
    
    
def combine_arrays_to_dataframe(arrays, column_names):
    """
    Combine multiple 2D NumPy arrays into a single Pandas DataFrame.

    Parameters:
    arrays (list of np.ndarray): List of 2D NumPy arrays to combine.
    column_names (list of str): List of column names for the resulting DataFrame.

    Returns:
    pd.DataFrame: A combined DataFrame with all arrays.
    """
    # Concatenate arrays column-wise
    combined_array = np.hstack(arrays)
    
    # Create DataFrame
    df = pd.DataFrame(combined_array, columns=column_names)
    
    return df
    
    
    

# Define column names for the combined DataFrame Stride Length
column_names = [f'base_{i}' for i in range(filtered_Len.shape[1])] + \
               [f'first_{i}' for i in range(filtered_Len2.shape[1])] + \
               [f'last_{i}' for i in range(filtered_Len3.shape[1])]

# Combine arrays into a single DataFrame
combined_Len = combine_arrays_to_dataframe(
    [filtered_Len, filtered_Len2, filtered_Len3],
    column_names
)
    
    
    
# Create a boxplot using seaborn for the data without outliers
sns.boxplot(data=combined_Len)

# Show the plot
plt.show()






# Define column names for the combined DataFrame Stride Time\
column_T_names = [f'base_{i}' for i in range(filtered_Time.shape[1])] + \
               [f'first_{i}' for i in range(filtered_Time2.shape[1])] + \
               [f'last_{i}' for i in range(filtered_Time3.shape[1])]

    

# Combine arrays into a single DataFrame
combined_Time = combine_arrays_to_dataframe(
    [filtered_Time, filtered_Time2, filtered_Time3],
    column_names
)




# Create a boxplot using seaborn for the data without outliers
sns.boxplot(data=combined_Time)

# Show the plot
plt.show()






# Define column names for the combined DataFrame Step Width
column_W_names = [f'base_{i}' for i in range(filtered_Width.shape[1])] + \
               [f'first_{i}' for i in range(filtered_Width2.shape[1])] + \
               [f'last_{i}' for i in range(filtered_Width3.shape[1])]

    

# Combine arrays into a single DataFrame
combined_Width = combine_arrays_to_dataframe(
    [filtered_Width, filtered_Width2, filtered_Width3],
    column_names
)




# Create a boxplot using seaborn for the data without outliers
sns.boxplot(data=combined_Width)

# Show the plot
plt.show()







def perform_repeated_measures_anova(df):
    """
    Perform a repeated measures ANOVA on the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with repeated measures data in long format.

    Returns:
    None: Prints the ANOVA results.
    """
    # Melt the DataFrame to long format
    long_df = df.melt(id_vars=None, var_name='Condition', value_name='Value')
    
    # Add an 'id' column to represent each row's subject
    long_df['Subject'] = long_df.groupby('Condition').cumcount()
    
    # Perform repeated measures ANOVA
    model = AnovaRM(long_df, 'Value', 'Subject', within=['Condition'])
    results = model.fit()
    
    print("\nRepeated Measures ANOVA Results:")
    print(results)





# Perform repeated measures ANOVA
perform_repeated_measures_anova(combined_Len)
perform_repeated_measures_anova(combined_Time)
perform_repeated_measures_anova(combined_Width)


#calcuate spatiotemporal averages and variability

Stride_LengthB = (Lft_StrL_B + Rgt_StrL_B)/2
Stride_Length1 = (Lft_StrL_first + Rgt_StrL_first)/2
Stride_Length2 = (Lft_StrL_last + Rgt_StrL_last)/2


Stride_TimeB = (Lft_StrT_B + Rgt_StrT_B)/2
Stride_Time1 = (Lft_StrT_first + Rgt_StrT_first)/2
Stride_Time2 = (Lft_StrT_last + Rgt_StrT_last)/2


Step_WidthB = (Lft_StepW_B + Rgt_StepW_B)/2
Step_Width1 = (Lft_StepW_first + Rgt_StepW_first)/2
Step_Width2 = (Lft_StepW_last + Rgt_StepW_last)/2

 
fil_StrideL, fil_StrideL2, fil_StrideL3 = filter_other_arrays([Stride_LengthB,Stride_Length1, Stride_Length2], removed_indices)
fil_StrideT, fil_StrideT2, filt_StrideT3 = filter_other_arrays([Stride_TimeB,Stride_Time1, Stride_Time2], removed_indices)
fil_StepW, fil_StepW2, fil_StepW3 = filter_other_arrays([Step_WidthB,Step_Width1, Step_Width2], removed_indices)



# Reshape the Data
fil_StrideL= fil_StrideL.reshape(-1, 1)  # Shape will be (16, 1)
fil_StrideL2= fil_StrideL2.reshape(-1,1)
fil_StrideL3 =fil_StrideL3.reshape(-1,1)

fil_StrideT = fil_StrideT.reshape(-1,1)
fil_StrideT2 = fil_StrideT2.reshape(-1,1)
fil_StrideT3 = filt_StrideT3.reshape(-1,1)

fil_StepW = fil_StepW.reshape(-1,1)
fil_StepW2 =  fil_StepW2.reshape(-1,1)
fil_StepW3 = fil_StepW3.reshape(-1,1)









Stride_Length_SD_B = (Lft_StrL_SD_B + Rgt_StrL_SD_B)/2
Stride_Length1_SD = (Lft_StrL_first_SD + Rgt_StrL_first_SD)/2
Stride_Length2_SD = (Lft_StrL_last_SD + Rgt_StrL_last_SD)/2


Stride_Time_SD_B = (Lft_StrT_SD_B + Rgt_StrT_SD_B)/2
Stride_Time1_SD = (Lft_StrT_first_SD + Rgt_StrT_first_SD)/2
Stride_Time2_SD = (Lft_StrT_last_SD + Rgt_StrT_last_SD)/2


Step_Width_SD_B = (Lft_StepW_SD_B + Rgt_StepW_SD_B)/2
Step_Width1_SD = (Lft_StepW_first_SD + Rgt_StepW_first_SD)/2
Step_Width2_SD = (Lft_StepW_last_SD + Rgt_StepW_last_SD)/2

 
fil_StrideL_SD, fil_StrideL2_SD, fil_StrideL3_SD = filter_other_arrays([Stride_Length_SD_B,Stride_Length1_SD, Stride_Length2_SD], removed_indices)
fil_StrideT_SD, fil_StrideT2_SD, filt_StrideT3_SD = filter_other_arrays([Stride_Time_SD_B,Stride_Time1_SD, Stride_Time2_SD], removed_indices)
fil_StepW_SD, fil_StepW2_SD, fil_StepW3_SD = filter_other_arrays([Step_Width_SD_B,Step_Width1_SD, Step_Width2_SD], removed_indices)



# Reshape the Data
fil_StrideL_SD= fil_StrideL_SD.reshape(-1, 1)  # Shape will be (16, 1)
fil_StrideL2_SD= fil_StrideL2_SD.reshape(-1,1)
fil_StrideL3_SD =fil_StrideL3_SD.reshape(-1,1)

fil_StrideT_SD = fil_StrideT_SD.reshape(-1,1)
fil_StrideT2_SD = fil_StrideT2_SD.reshape(-1,1)
fil_StrideT3_SD = filt_StrideT3_SD.reshape(-1,1)

fil_StepW_SD = fil_StepW_SD.reshape(-1,1)
fil_StepW2_SD =  fil_StepW2_SD.reshape(-1,1)
fil_StepW3_SD = fil_StepW3_SD.reshape(-1,1)



















# Define column names for the combined DataFrame Average Stride Length
column_StrideL_names = [f'base_{i}' for i in range(fil_StrideL.shape[1])] + \
               [f'first_{i}' for i in range(fil_StrideL2.shape[1])] + \
               [f'last_{i}' for i in range(fil_StrideL3.shape[1])]

    

# Combine arrays into a single DataFrame
combined_StrideLen = combine_arrays_to_dataframe(
    [fil_StrideL, fil_StrideL2, fil_StrideL3],
    column_names
)




# Create a boxplot using seaborn for the data without outliers
sns.boxplot(data=combined_StrideLen)

# Show the plot
plt.show()







# Define column names for the combined DataFrame Average Stride Time
column_StrideT_names = [f'base_{i}' for i in range(fil_StrideT.shape[1])] + \
               [f'first_{i}' for i in range(fil_StrideT2.shape[1])] + \
               [f'last_{i}' for i in range(fil_StrideT3.shape[1])]

    

# Combine arrays into a single DataFrame
combined_StrideTime = combine_arrays_to_dataframe(
    [fil_StrideT, fil_StrideT2, fil_StrideT3],
    column_names
)




# Create a boxplot using seaborn for the data without outliers
sns.boxplot(data=combined_StrideTime)

# Show the plot
plt.show()






# Define column names for the combined DataFrame Average Step Width
column_StepW_names = [f'base_{i}' for i in range(fil_StepW.shape[1])] + \
               [f'first_{i}' for i in range(fil_StepW2.shape[1])] + \
               [f'last_{i}' for i in range(fil_StepW3.shape[1])]

    

# Combine arrays into a single DataFrame
combined_StepW = combine_arrays_to_dataframe(
    [fil_StepW, fil_StepW2, fil_StepW3],
    column_names
)




# Create a boxplot using seaborn for the data without outliers
sns.boxplot(data=combined_StepW)

# Show the plot
plt.show()

















# Define column names for the combined DataFrame Average Stride Length
column_StrideL_SD_names = [f'base_{i}' for i in range(fil_StrideL_SD.shape[1])] + \
               [f'first_{i}' for i in range(fil_StrideL2_SD.shape[1])] + \
               [f'last_{i}' for i in range(fil_StrideL3_SD.shape[1])]

    

# Combine arrays into a single DataFrame
combined_StrideLen_SD= combine_arrays_to_dataframe(
    [fil_StrideL_SD, fil_StrideL2_SD, fil_StrideL3_SD],
    column_names
)




# Create a boxplot using seaborn for the data without outliers
sns.boxplot(data=combined_StrideLen_SD)

# Show the plot
plt.show()







# Define column names for the combined DataFrame Average Stride Time
column_StrideT_SD_names = [f'base_{i}' for i in range(fil_StrideT_SD.shape[1])] + \
               [f'first_{i}' for i in range(fil_StrideT2_SD.shape[1])] + \
               [f'last_{i}' for i in range(fil_StrideT3_SD.shape[1])]

    

# Combine arrays into a single DataFrame
combined_StrideTime_SD = combine_arrays_to_dataframe(
    [fil_StrideT_SD, fil_StrideT2_SD, fil_StrideT3_SD],
    column_names
)




# Create a boxplot using seaborn for the data without outliers
sns.boxplot(data=combined_StrideTime_SD)

# Show the plot
plt.show()






# Define column names for the combined DataFrame Average Step Width
column_StepW_SD_names = [f'base_{i}' for i in range(fil_StepW_SD.shape[1])] + \
               [f'first_{i}' for i in range(fil_StepW2_SD.shape[1])] + \
               [f'last_{i}' for i in range(fil_StepW3_SD.shape[1])]

    

# Combine arrays into a single DataFrame
combined_StepW_SD = combine_arrays_to_dataframe(
    [fil_StepW_SD, fil_StepW2_SD, fil_StepW3_SD],
    column_names
)




# Create a boxplot using seaborn for the data without outliers
sns.boxplot(data=combined_StepW_SD)

# Show the plot
plt.show()







# Perform repeated measures ANOVA
perform_repeated_measures_anova(combined_StrideLen)
perform_repeated_measures_anova(combined_StrideTime)
perform_repeated_measures_anova(combined_StepW)



perform_repeated_measures_anova(combined_StrideLen_SD)
perform_repeated_measures_anova(combined_StrideTime_SD)
perform_repeated_measures_anova(combined_StepW_SD)



# Calculate Phase Coordination Index #




Left_Stp_T = 'L_StepTime'
Left_Stp_L = 'L_StepLength'




Right_Stp_T = 'R_StepTime'
Right_Stp_L = 'R_StepLength'




# Extract the specific parameter from each nested dictionary
Left_StepT_first = {k: v[Left_Stp_T] for k, v in first.items()}
Left_StepL_first = {k: v[Left_Stp_L] for k, v in first.items()}
    

# Extract the specific parameter from each nested dictionary
Right_StepT_first = {k: v[Right_Stp_T] for k, v in first.items()}
Right_StepL_first = {k: v[Right_Stp_L] for k, v in first.items()}



# Extract the specific parameter from each nested dictionary
Left_StepT_last = {k: v[Left_Stp_T] for k, v in last.items()}
Left_StepL_last = {k: v[Left_Stp_L] for k, v in last.items()}

# Extract the specific parameter from each nested dictionary
Right_StepT_last = {k: v[Right_Stp_T] for k, v in last.items()}
Right_StepL_last = {k: v[Right_Stp_L] for k, v in last.items()}
    








# Convert left values from dictionary to numpy array
values = Left_StepT_first.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StpT_first = np.array(values_list)


values = Left_StepL_first.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StpL_first  = np.array(values_list)





values = Left_StepT_last.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StpT_last = np.array(values_list)


values = Left_StepL_last.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StpL_last  = np.array(values_list)







# Convert right values from dictionary to numpy array
values = Right_StepT_first.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StpT_first = np.array(values_list)


values = Right_StepL_first.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StpL_first  = np.array(values_list)


values = Right_StepT_last.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StpT_last = np.array(values_list)


values = Right_StepL_last.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StpL_last  = np.array(values_list)





# Extract the specific parameter from each nested dictionary
Left_StpT_B = {k: v[Left_Stp_T] for k, v in B_dataframes.items()}
Left_StpL_B = {k: v[Left_Stp_L] for k, v in B_dataframes.items()}
    

# Extract the specific parameter from each nested dictionary
Right_StpT_B = {k: v[Right_Stp_T] for k, v in B_dataframes.items()}
Right_StpL_B = {k: v[Right_Stp_L] for k, v in B_dataframes.items()}
    



    


values = Left_StpT_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StpT_B = np.array(values_list)


values = Left_StpL_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Lft_StpL_B  = np.array(values_list)




values = Right_StpT_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StpT_B = np.array(values_list)


values = Right_StpL_B.values()
# Convert the values to a list
values_list = list(values)
# Convert the list of values to a NumPy array
Rgt_StpL_B  = np.array(values_list)



fil_RstpT_B, fil_RstpT_B_first, fil_RstpT_B_last = filter_other_arrays([Rgt_StpT_B,Rgt_StpT_first, Rgt_StpT_last], removed_indices)
fil_LstpT_B, fil_LstpT_B_first, fil_LstpT_B_last = filter_other_arrays([Lft_StpT_B,Lft_StpT_first, Lft_StpT_last], removed_indices)



fil_RstpT_B= fil_RstpT_B.reshape(-1, 1)
fil_LstpT_B = fil_LstpT_B.reshape(-1, 1)
fil_RstpT_first = fil_RstpT_B_first.reshape(-1, 1)
fil_LstpT_first = fil_LstpT_B_first.reshape(-1, 1)
fil_RstpT_last = fil_RstpT_B_last.reshape(-1, 1)
fil_LstpT_last = fil_LstpT_B_last.reshape(-1, 1)


#PCI 
PCI_t = 360*(np.absolute(fil_RstpT_B - fil_LstpT_B)/fil_StrideT)
PCI_B = np.absolute(PCI_t-180)


PCI_t_first = 360*(np.absolute(fil_RstpT_first - fil_LstpT_first)/fil_StrideT2)
PCI_first = np.absolute(PCI_t_first-180)


PCI_t_last = 360*(np.absolute(fil_RstpT_last - fil_LstpT_last)/fil_StrideT3)
PCI_last = np.absolute(PCI_t_last-180)






# Define column names for the combined DataFrame Average Stride Time
column_PCI_names = [f'base_{i}' for i in range(PCI_B.shape[1])] + \
               [f'first_{i}' for i in range(PCI_first.shape[1])] + \
               [f'last_{i}' for i in range(PCI_last.shape[1])]

    

# Combine arrays into a single DataFrame
combined_PCI = combine_arrays_to_dataframe(
    [PCI_B, PCI_first, PCI_last],
    column_names
)




# Create a boxplot using seaborn for the data without outliers
sns.boxplot(data=combined_PCI)

# Show the plot
plt.show()

# PCI ANOVA
perform_repeated_measures_anova(combined_PCI)
