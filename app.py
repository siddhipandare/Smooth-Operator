# Set up a simple Flask app
from flask import Flask, request
# Import render_template to render HTML templates
from flask import render_template
import json

from utils import fetch_time_series, get_all_datasets, get_entropy_series, get_snr_series

filename_to_type = {
    'taxi.csv': 'csv',
    'temp.csv': 'csv',
    'stock_prices.csv': 'csv',
    'nz_tourist_monthly.json': 'json',
    'eeg_chan20_2500.json': 'json',
    'chi_homicide_monthly.json': 'json',
    'stock_tsla_price.json': 'json',
    'power.csv': 'csv',
    'sine.csv': 'csv',
}

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/visualize/<filename>')
def visualize(filename):
    time_series = fetch_time_series(filename, type=filename_to_type[filename])
    dataset_list = get_all_datasets()
    return render_template('visualize.html', time_series=time_series, dataset_list=dataset_list)

# Create a new route to handle POST requests
@app.route('/fetch_running_snr', methods=['POST'])
def fetch_running_snr():
    # Get the time series from the request
    time_series = request.get_json()['time_series']
    # Get the window length from the request
    window_length = request.get_json()['window_length']

    # Get the running snr list
    running_snr_list = get_snr_series(time_series, window_length)

    # Return the running snr list as a JSON response
    return json.dumps({'snr_time_series': running_snr_list})

@app.route('/fetch_running_entropy', methods=['POST'])
def fetch_running_entropy():
    # Get the time series from the request
    time_series = request.get_json()['time_series']
    # Get the window length from the request
    window_length = request.get_json()['window_length']

    # Get the running entropy list
    running_entropy_list = get_entropy_series(time_series, window_length)

    # Return the running snr list as a JSON response
    return json.dumps({"entropy_time_series": running_entropy_list})

if __name__ == '__main__':
    # Run on port 5000 with debug mode
    app.run(port=5000, debug=True)