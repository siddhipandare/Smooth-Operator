# 8803-MDS-Project

### Structure of the Repository:
1. **app.py**: This is the main file that runs the application.
The main routes of interest are :
    1. **/visualize/<filename.csv>:** This helps visualize a particular dataset with different levels of smoothing with a simple moving average. Eg. http://localhost:5000/visualize/taxi.csv. Here, the orange line is the running signal-to-noise ratio and the green line is the running information entropy of the dataset.
    2. **/user_study_1:** Here, you can choose a dataset, a statistical measure, and a smoothing function and click on Update button to see the results. The png of the plot will be downloaded for you as well.

2. **utils.py:** This file contains all the helper functions that are used in app.py
3. **8803-MDS-Project:** This folder contains all the notebooks used in the project and the results.

### Inside the 8803-MDS-Project folder:
1. **datasets**: This folder contains all the datasets used in the project.
2. **datasets_user_study_1:** This folder contains the datasets used in the user study in the project. They are arranged in the category folders.
3. **heatmaps:** This folder contains the png files of the heatmaps generated from the results of the user study. They were generated using the user_study_analysis.ipynb notebook.
4. **pdfs_userstudy_numbered:** This folder contains the pdfs of the user study questions. In each pdf, the options are numbered and the user study participants were asked to choose the option number that they thought was in the top 3.
5. **mappings:** This folder contains the mappings of the options in the user study to the actual smoothing techniques used for that option. This helps us to analyze the results of the user study.
6. **pdfs_annotated:** This folder contains the pdfs of the user study questions with the options labeled with the smoothing techniques used for that option. This is for our ease of analysis.
7. **results:** This folder contains the png files of all the plots generated for the time series using the various smoothing techniques.
8. **generate_pdfs.py:** This script constructs the pdfs to be used in the user study.
9. **entropy_data.txt, exp_smoothing_data.txt, kurtosis_data.txt, low_pass_kurtosis_data.txt:** All the txt files used to generate the plots in the qualitative and quantitative analysis sections of the paper.
10. **mynb.ipynb:** This notebook contains the code for generating the plots in the qualitative and quantitative analysis sections of the paper.
11. **8803-MDS User Study file.csv:** The results of the user study (exported from google forms as a csv file)
12. **user_study_analysis.ipynb:** This notebook contains the code for analyzing the results of the user study.
13. **feature_importance.ipynb:** This notebook contains the code for generating the feature importance plots for the datasets used in the user study, as well as analysing the time series using seasonal decomposition and the autocorrelation plots.
14. **Entropy_SNR.ipynb:** This notebook contains the code we used for experimentation with different statistical measures to use in our project.
15. **different_approach.ipynb:** This notebook contains the code we used for running our analysis on the running SNR / running entropy plots, a technique we couldn't explore further due to lack of time.
    
## Steps to Reproduce the Results:

To get the code from the source, clone this repository:
```bash
git clone https://github.gatech.edu/ssangle6/8803-MDS-Project.git
```
Open the repository and run the `app.py` file to access the web application.

```bash
cd 8803-MDS-Project
python app.py
```

**NOTE:** Install flask before running the app using 
```bash
pip install flask
```

On a browser open http://localhost:5000/user_study_1. At the interface choose the desired dataset, smoothing technique and statistical measure from the drop down menu and press `Update`. The smoothened plot along with best smoothing length and oversmoothing threshold will be displayed on the screen. The plot will also be downloaded in the `.png` format. It might take some time to generate some results due to the size of the datasets so please be patient. 
The following snapshot captures our interface smoothing the *nz_tourist_monthly.json* dataset with a combination of gaussian and kurtosis.

<p align="center">
    <img src= "https://github.gatech.edu/storage/user/66957/files/6bd43fee-4043-49d4-bc3f-632be9e03380">
</p>


To generate and download all the smoothened plots for all the datasets in a quicker manner, click `Generate`. This will display buttons for all the 15 combinations of all the 12 datasets (15 * 12 buttons). The desired combination and dataset can then be chosen from here. 
