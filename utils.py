import pandas as pd
import numpy as np
import os
import glob
import json
import antropy as ant

BASE_DIR = './8803-MDS-Project/datasets'

def fetch_time_series(filename, type):
    '''
    Fetch time series data from file depending on type
    '''
    if type == 'csv':
        dataframe = pd.read_csv(os.path.join(BASE_DIR, filename))
        # Get 'value' column
        time_series = dataframe['value'].values
    elif type == 'json':
        with open(os.path.join(BASE_DIR, filename)) as f:
            data = json.load(f)
        time_series = data
    else:
        raise ValueError('Invalid type')

    # Z-normalize time series
    time_series = (time_series - np.mean(time_series)) / np.std(time_series)

    # Return time series as a python list
    return time_series.tolist()
    
def get_all_datasets():
    '''
    Get all datasets from the datasets folder
    '''
    full_paths = glob.glob(os.path.join(BASE_DIR, '*'))
    filenames = [os.path.basename(path) for path in full_paths]
    return filenames
    
def new_approach(time_series):
    '''
    New approach to time series smoothing
    '''
    # Convert to numpy array
    time_series = np.array(time_series)
    window_length = int(0.1 * len(time_series))
    snr_list = np.array([])

    for i in range(0, len(time_series) - window_length):
        window = time_series[i:i+window_length]
        snr = np.mean(window) / np.std(window)
        snr **= 2
        snr_list = np.append(snr_list, snr)
    
    # Scale the snr_list to fit between 0 and 3
    snr_list = (snr_list - np.min(snr_list)) / (np.max(snr_list) - np.min(snr_list)) * 3

    total_area = np.trapz(snr_list)

    # Find partitions in the snr_list such that area under the curve for each partition is equal to total_area / 5
    no_of_partitions = 6
    partition = np.array([])
    left = 0
    right = 0
    while right < len(snr_list):
        if np.trapz(snr_list[left:right]) > total_area / no_of_partitions:
            partition = np.append(partition, right)
            left = right
        right += 1

    # Return the snr_list and the partitions
    return snr_list.tolist(), partition.tolist()

def new_approach_app_ent(time_series):
    '''
    New approach to time series smoothing with app ent
    '''
    # Convert to numpy array
    time_series = np.array(time_series)
    window_length = int(0.1 * len(time_series))
    app_ent_list = np.array([])

    for i in range(0, len(time_series) - window_length):
        window = time_series[i:i+window_length]
        app_ent = ant.app_entropy(window, 2)
        app_ent_list = np.append(app_ent_list, app_ent)
    
    # Scale the app_ent_list to fit between 0 and 3
    app_ent_list = (app_ent_list - np.min(app_ent_list)) / (np.max(app_ent_list) - np.min(app_ent_list)) * 3

    total_area = np.trapz(app_ent_list)

    # Find partitions in the app_ent_list such that area under the curve for each partition is equal to total_area / 5
    no_of_partitions = 6
    partition = np.array([])
    left = 0
    right = 0
    while right < len(app_ent_list):
        if np.trapz(app_ent_list[left:right]) > total_area / no_of_partitions:
            partition = np.append(partition, right)
            left = right
        right += 1

    # Return the app_ent_list and the partitions
    return app_ent_list.tolist(), partition.tolist()
