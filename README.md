# Credit_Risk_Analysis

## Overview

The purpose of this project is to understand how to utilize Machine Learning algorithms to make predictions based on the data patterns provided. In this challenge, we focus on supervised learning using a free dataset from LendingClub, a P2P lending service company to evaluate and predict credit risk. In this analysis, we use different Machine Learning techniques to train and evaluate the data with unbalanced classes. The dataset from LendingClub has a disproportionate number of good loans to the amount of risky loans. To balance out the classifications, allow for more meaningful predictions, and improve the accuracy score, various Machine Learning algorithms are used to resample the data. These algorithms include:
* RandomOverSampler  
* SMOTE  
* ClusterCentroids  
* SMOTEENN  
* BalancedRandomForestClassifier  
* EasyEnsembleClassifier

### Tools used in this project

* Jupyter Notebook

# Results

To balance out the data, resampling is done using Python's **scikit-learn** and **imbalanced-learn** libraries to evaluate the results and provide a comparison for the analysis.

The original dataset contained 115,675 loan applications in Q1 of 2019. After dropping the null data and reclassifying the applicants to either "low risk" or "high risk", the dataset to shrunk down to 68,817 total applications with 99.5% classified as "low risk". Using a 75/25% split for training vs testing data would result in 51,366 "low risk" and 246 "high risk" applications in the training set.
![Ss1](https://user-images.githubusercontent.com/109183214/206361943-1bdba58c-ce84-41a3-94ae-ce5dd61aff8a.png)



