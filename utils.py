import pandas as pd
import numpy as np
import os
import glob
import json

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
    
def get_entropy_series(time_series, window_length):
    '''
    Time series: a list of values
    window_length: length of running window to be used
    '''
    time_series = np.array(time_series)
    entropy_series = np.array([])

    for i in range(len(time_series) - window_length):
        # Get deep copy of the window
        window = time_series[i:i+window_length].copy()

        # Get span of the time series and discretize values to 10 bins
        bins = np.linspace(np.min(time_series), np.max(time_series), 30)

        # Get histogram of the window
        hist, _ = np.histogram(window, bins=bins)

        # Add 1 to each bin to avoid division by zero
        hist = hist + 1

        # Get probability distribution of the window
        prob_dist = hist / np.sum(hist)

        # Get entropy of the window
        entropy = -np.sum(prob_dist * np.log2(prob_dist))

        # Append the entropy to the entropy series
        entropy_series = np.append(entropy_series, entropy)

    # Scale the entropy series to fit between 0 and 3
    entropy_series = (entropy_series - np.min(entropy_series)) / (np.max(entropy_series) - np.min(entropy_series)) * 3
    return entropy_series.tolist()

def get_snr_series(time_series, window_length):
    '''
    Time series: a list of values
    window_length: length of running window to be used
    '''
    time_series = np.array(time_series)
    snr_series = np.array([])
    for i in range(len(time_series) - window_length):
        # Get deep copy of the window
        window = time_series[i:i+window_length].copy()

        # Get standard deviation of the window
        std = np.std(window)

        # Get mean of the window
        mean = np.mean(window)

        # Get signal to noise ratio of the window
        snr = mean / std

        # Square the signal to noise ratio
        snr = snr ** 2

        # Append the signal to noise ratio to the signal to noise ratio series
        snr_series = np.append(snr_series, snr)

    # Scale the signal to noise ratio series to fit between 0 and 3
    snr_series = (snr_series - np.min(snr_series)) / (np.max(snr_series) - np.min(snr_series)) * 3
    return snr_series.tolist()