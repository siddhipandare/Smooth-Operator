import pandas as pd
import numpy as np
import os
import glob
import json

BASE_DIR = './8803-MDS-Project/datasets'
USER_STUDY_1_DIR = './8803-MDS-Project/datasets_user_study_1'

smoothing_techniques = [
    'mean',
    'median',
    'gaussian',
    'exponential_smoothing',
    'low_pass_filter'
]

statistical_measures = [
    'kurtosis',
    'signal to noise ratio',
    'entropy'
]

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

def get_all_datasets_user_study_1():
    '''
    Get all datasets from the datasets_user_study_1 folder
    '''
    # Set full paths as all datasets within all subfolders of USER_STUDY_1_DIR
    full_paths = glob.glob(os.path.join(USER_STUDY_1_DIR, '**', '*'), recursive=True)
    # Remove from full_paths all files not ending with .csv or .json
    full_paths = [path for path in full_paths if path.endswith('.csv') or path.endswith('.json')]
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

def get_detrended_time_series(time_series, window_length):
    time_series = np.array(time_series)
    sma_series = np.array([])

    for i in range(len(time_series) - window_length):
        # Get a deep copy of the window
        window = time_series[i:i+window_length].copy()
        
        # Get the sma of the window
        sma = np.mean(window)

        # Append the sma to the detrended time series
        sma_series = np.append(sma_series, sma)

    # Now, subtract the detrended time series from the original time series
    detrended_time_series = time_series[window_length//2:-window_length//2] - sma_series

    # Return the detrende time series and its variance
    return detrended_time_series.tolist(), np.var(detrended_time_series)

def kurtosis(time_series):
    # Get the kurtosis of the time series using numpy
    time_series = np.array(time_series)
    mean = np.mean(time_series)
    std = np.std(time_series)
    kurtosis = np.mean((time_series - mean) ** 4) / std ** 4

    return kurtosis

def signal_to_noise_ratio(time_series):
    # Get the signal to noise ratio of the time series using numpy
    time_series = np.array(time_series)
    mean = np.mean(time_series)
    std = np.std(time_series)
    snr = mean / std

    snr = snr ** 2

    return snr

def entropy(time_series):
    # Get the entropy of the time series using numpy
    time_series = np.array(time_series)
    bins = np.linspace(np.min(time_series), np.max(time_series), 30)
    hist, _ = np.histogram(time_series, bins=bins)
    hist = hist + 1
    prob_dist = hist / np.sum(hist)
    entropy = -np.sum(prob_dist * np.log2(prob_dist))

    return entropy

statistical_measure_to_function = {
    'kurtosis': kurtosis,
    'signal to noise ratio': signal_to_noise_ratio,
    'entropy': entropy
}

def smooth_mean(time_series, window_length):
    time_series = np.array(time_series)
    smoothed_series = np.array([])

    for i in range(len(time_series) - window_length):
        # Get a deep copy of the window
        window = time_series[i:i+window_length].copy()
        
        # Get the mean of the window
        mean = np.mean(window)

        # Append the mean to the smoothed time series
        smoothed_series = np.append(smoothed_series, mean)

    return smoothed_series.tolist()

def smooth_median(time_series, window_length):
    time_series = np.array(time_series)
    smoothed_series = np.array([])

    for i in range(len(time_series) - window_length):
        # Get a deep copy of the window
        window = time_series[i:i+window_length].copy()
        
        # Get the median of the window
        median = np.median(window)

        # Append the median to the smoothed time series
        smoothed_series = np.append(smoothed_series, median)

    return smoothed_series.tolist()

def gaussian_kernel(window_length, sigma):
    # Get the gaussian kernel
    x = np.linspace(-window_length//2, window_length//2, window_length)
    kernel = np.exp(-x**2 / (2 * sigma**2))

    # Normalize the kernel
    kernel = kernel / np.sum(kernel)

    return kernel

def smooth_gaussian(time_series, window_length):
    '''
    time_series: a list of values
    window_length: length of gauusian kernel to be used
    '''
    time_series = np.array(time_series)
    smoothed_series = np.array([])
    kernel = gaussian_kernel(window_length, sigma=window_length//2)

    # Convolve the time series with the kernel
    smoothed_series = np.convolve(time_series, kernel, mode='same')

    return smoothed_series.tolist()

def smooth_exponential(time_series, window_length):
    # Use the exponential moving average to smooth the time series
    time_series = np.array(time_series)
    exponential_smoothed_series = np.array([])
    alpha = 2 / (window_length + 1)

    # Exponential moving average
    for i in range(len(time_series)):
        if i == 0:
            exponential_smoothed_series = np.append(exponential_smoothed_series, time_series[i])
        else:
            exponential_smoothed_series = np.append(exponential_smoothed_series, alpha * time_series[i] + (1 - alpha) * exponential_smoothed_series[i-1])

    return exponential_smoothed_series.tolist()

def smooth_low_pass(time_series, window_length):
    # Use a low pass filter to smooth the time series
    time_series = np.array(time_series)
    low_pass_smoothed_series = np.array([])

    # Low pass filter
    for i in range(len(time_series)):
        if i == 0:
            low_pass_smoothed_series = np.append(low_pass_smoothed_series, time_series[i])
        else:
            low_pass_smoothed_series = np.append(low_pass_smoothed_series, 0.5 * time_series[i] + 0.5 * low_pass_smoothed_series[i-1])

    return low_pass_smoothed_series.tolist()

smoothing_technique_to_function = {
    'mean': smooth_mean,
    'median': smooth_median,
    'gaussian': smooth_gaussian,
    'exponential_smoothing': smooth_exponential,
    'low_pass_filter': smooth_low_pass
}

statistical_measure_to_comparator_function = {
    'kurtosis': lambda x, y: x > y,
    'signal to noise ratio': lambda x, y: x > y,
    'entropy': lambda x, y: x < y
}



def process(time_series, smoothing_technique, statistical_measure):
    '''
    time_series: a list of values
    smoothing_technique: one of the smoothing techniques
    statistical_measure: one of the statistical measures
    '''

    len_time_series = len(time_series)

    # Check all window sizes from 1 to 30% of the length of the time series
    # and keep track of the window size that gives the best result

    best_window_size = 1
    best_statistical_measure = statistical_measure_to_function[statistical_measure](time_series)

    for window_size in range(2, int(len_time_series * 0.3)):
        # Smooth the time series
        smoothed_time_series = smoothing_technique_to_function[smoothing_technique](time_series, window_size)

        # Get the statistical measure of the smoothed time series
        smoothed_time_series_statistical_measure = statistical_measure_to_function[statistical_measure](smoothed_time_series)

        # Check if the statistical measure of the smoothed time series is better than the best one
        if statistical_measure_to_comparator_function[statistical_measure](smoothed_time_series_statistical_measure, best_statistical_measure):
            best_window_size = window_size
            best_statistical_measure = smoothed_time_series_statistical_measure

    # Smooth the time series with the best window size
    smoothed_time_series = smoothing_technique_to_function[smoothing_technique](time_series, best_window_size)

    return smoothed_time_series, best_window_size

        