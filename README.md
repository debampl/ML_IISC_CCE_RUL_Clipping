# IISC CCE Machine Learning and Artificial Intelligence with Python Project

<img src="https://github.com/VijayAerospace/ML_IISC_CCE_RUL_FD003_V1_2023/raw/main/Image%2021-04-23%20at%2010.54%20PM.jpeg" alt="Schematic Diagram of Aircraft Engine" width="600">
Simplified diagram of engine simulated in C-MAPSS is shown above. 


**Objective:** The objective of the project is to develop a machine learning model that can predict the number of remaining useful life before failure for each engine in the test set. This involves analyzing the training data to identify patterns in the sensor measurements and operational settings that are associated with engine failure, and using this information to make accurate predictions for the test set.

**Data Set:** FD003. (Train, Test, RUL data) Train trajectories: 100 Test trajectories: 100 Conditions: ONE (Sea Level) Fault Modes: TWO (HPC Degradation, Fan Degradation)

**Team members:**

- Mr. Vijay Kothari (Project Leader)
- Dr. Debabrata Adhikari
- Dr. Keshava Kumar

**Date:** 22nd April 2023

**GitHub link:** https://github.com/debampl/ML_IISC_CCE_RUL_Clipping

**Code name:** ML_IISC_CCE_RUL_FD003_V1_2023.ipynb

**Reference:**

- Data source: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
- Research Paper: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
- Kaggle: https://www.kaggle.com/code/wassimderbel/nasa-predictive-maintenance-rul
- Lecture notes: https://www.zenteiq.com/

**Libraries Used**

The following Python libraries were used in this project:

- pandas (imported as pd)
- numpy (imported as np)
- matplotlib.pyplot (imported as plt)
- seaborn (imported as sns)
- SelectFromModel from sklearn.feature_selection
- MinMaxScaler and StandardScaler from sklearn.preprocessing
- mean_squared_error and r2_score from sklearn.metrics
- make_pipeline and Pipeline from sklearn.pipeline
- LinearRegression and Lasso from sklearn.linear_model
- PolynomialFeatures from sklearn.preprocessing
- KNeighborsRegressor from sklearn.neighbors
- SVR from sklearn.svm
- RandomForestRegressor, GradientBoostingRegressor, and VotingRegressor from sklearn.ensemble
- GridSearchCV from sklearn.model_selection

These libraries were used for various tasks such as data analysis, feature selection, data preprocessing, model training, model evaluation, and hyperparameter tuning.

**Final Deliverables:**

- ML_IISC_CCE_RUL_FD003_V1_2023.pdf: This is a presentation report summarizing the project findings, methodology, results, and conclusions.
- ML_IISC_CCE_RUL_FD003_V1_2023.ipynb: This is a Jupyter Notebook file (.ipynb) containing the code for the project, including data analysis, model training, and evaluation. It also includes comments and explanations to make it easy to understand (wherever feasible and required).
- ML_IISC_CCE_RUL_FD003_V1_2023_IEEE.pdf: This is the IEEE


Project Workflow:

- Clone the GitHub repository from the provided link.
- Import the required Python libraries as mentioned in the "Libraries Used" section.
- Open the Jupyter Notebook file "ML_IISC_CCE_RUL_FD003_V1_2023.ipynb" in a Jupyter Notebook editor or any compatible IDE.
- Follow the comments and explanations provided in the code to understand the different steps involved in the project, including data analysis, feature selection, data preprocessing, model training, model evaluation, and hyperparameter tuning.
- Run the code cells sequentially to execute the tasks and see the results.
- Once the code execution is complete, you will get the predicted results for the remaining useful life (RUL) of the aircraft engines in the test set.
- Refer to the ML_IISC_CCE_RUL_FD003_V1_2023.pdf report for a summary of the project findings, methodology, results, and conclusions.
