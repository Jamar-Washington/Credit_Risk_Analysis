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

### Oversampling

The first model to balance the data is the **RandomOverSampler Model**. This model randomly selects from the minority class and adds it to the training set until both classifications are equal. In this case, 51,366 records will be in High Risk and Low Risk.

![randov](https://user-images.githubusercontent.com/109183214/206362760-9c253ab7-958e-43ec-8155-f9f4cd1839ef.png)

  * Balanced Accuracy score: 65.3%
  * "High Risk": precision rate - 1%, recall rate - 75%
  * "Low Risk:"  precision rate - 100%, recall rate - 62%  
  

The other model that uses oversampling is the **Synthetic Minority Oversampling Technique Model**. SMOTE is similar to the model above as it increases the size of the minority class. However, it does so by creating new values based on the value of the closest neighbors to the minority class instead of random selection. 

![SMOTE](https://user-images.githubusercontent.com/109183214/206363690-81b07a6e-a511-46cf-812f-75862c33921a.png)

  * Balanced Accuracy score: 63.2%.
  * "High Risk": precision rate - 1%, recall rate - 60%
  * "Low Risk:"  precision rate - 100%, recall rate - 66%

### Undersampling

The next model is an undersampling model called the **ClusterCentroids Model**. This model identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. For this model, there are 246 classified records each as High Risk and Low Risk.

![clustercent](https://user-images.githubusercontent.com/109183214/206364454-0ad79943-9b0f-4555-9ebe-cc19a0a0ccf2.png)

  * Balanced Accuracy score - 53.1%.
  * "High Risk": precision rate - 1%, recall rate - 67%
  * "Low Risk:"  precision rate - 100%, recall rate - 39%


### Combination Sampling

**(Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model** (SMOTEENN) combines aspects of both oversampling and undersampling.

![combo](https://user-images.githubusercontent.com/109183214/206365336-ecb80fd4-e898-4272-b222-8cacf86527a0.png)

  * Balanced Accuracy score - 64.9%.
  * "High Risk": precision rate - 1%, recall rate - 74%
  * "Low Risk:"  precision rate - 100%, recall rate - 56% 


