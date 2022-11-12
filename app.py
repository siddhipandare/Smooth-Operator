# Set up a simple Flask app
from flask import Flask
# Import render_template to render HTML templates
from flask import render_template

from utils import fetch_time_series, get_all_datasets, new_approach

filename_to_type = {
    'taxi.csv': 'csv',
    'temp.csv': 'csv',
    'stock_prices.csv': 'csv',
    'nz_tourist_monthly.json': 'json',
    'eeg_chan20_2500.json': 'json',
    'chi_homicide_monthly.json': 'json',
    'stock_tsla_price.json': 'json',
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

@app.route('/analyze/<filename>')
def analyze(filename):
    time_series = fetch_time_series(filename, type=filename_to_type[filename])
    snr_series, partitions = new_approach(time_series)
    dataset_list = get_all_datasets()
    return render_template('analyze.html', time_series=time_series, partitions=partitions, dataset_list=dataset_list, snr_series=snr_series)



if __name__ == '__main__':
    # Run on port 5000 with debug mode
    app.run(port=5000, debug=True)