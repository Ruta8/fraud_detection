# Fraud Detection Take Home Challenge

Highly imbalanced binary classification problem with messy data. The goal is to predict possibly high-value fraudulent transactions. Feature engineering was centred around time-aware RFM-style features. The model of choice was Random Forest for great performance, moderate interpretability, ability to deal with the large dimensionality. Loss function modified to take into account fraud value. The model was calibrated using Beta Calibration.

## Folder Structure

```
├── analysis.ipynb <- A Jupyter notebook with exploratory data analysis and model results analysis.             
├── data
│   ├── external                    <- Data from third party sources.
│   ├── processed                   <- The final, canonical data sets for modeling.
│   └── raw                         <- The original, immutable data dump.
├── fraud_detector                  
|   ├── data_cleaning               <- Scripts to clean the data. 
|   ├── feature_engineering         <- Scripts to turn cleaned data into features for modeling.
|   ├── modeling                    <- Scripts to train models and then use trained models to make predictions.
├── results                         <- Stores model predictions along with results of evaluation metrics.
├── main.py                         <- Runs scrips in fraud_detector.
├── README.md                       <- The top-level README for developers using this project.
```

## Project Challenges

* Class imbalance. Transaction data contain much more legitimate than fraudulent transactions: the percentage of fraudulent transactions is under 1%. 
* Concept drift. Transaction and fraud patterns change over time. On the one hand, the spending habits of credit card users are different during weekdays, weekends, vacation periods, and more generally evolve.
* Categorical features. Transactional data contain categorical features of high cardinality, such as the account number, merchant id, and postcode area.
* Sequential data. Each account number has generated a stream of sequential data with unique characteristics. A significant challenge was to use these streams in modeling to characterize the regular and irregular behaviors better.
* Messy data. As a challenge, the data contains many messy values. For example, merchant zip code columns have these values and more: *, ..., XXXXX, DUMMY.
* Performance measures. Standard measures for classification systems, such as the AUC ROC, are not well suited in this case because of the class imbalance issue and the complex cost structure of fraud detection. 
* The need to optimize money saved. I had to find a way to address the monetary value of fraudulent transactions to optimize the money saved.

## Feature Engineering

In this project, I implemented three types of feature transformation that are known to be relevant for payment card fraud detection:
* Binary encoding or one-hot encoding
* RMF (recency, frequency, monetary value)

The first type of transformation involves the date/time variable and consists in creating binary features that characterize potentially relevant periods (e.g. transaction_during_weekend).

The second type of transformation involved the account_number column and features that characterize the account spending behaviors. Here I follow the RFM (Recency, Frequency, Monetary value) framework proposed by [Vlasselaer (2015)](https://www.sciencedirect.com/science/article/abs/pii/S0167923615000846)

The third type of transformation involves the merchant_id column and creates new features that characterize the 'risk' associated with the merchant. 

## Model Choice
There are lots of high cardinality categorical features in this data set, leading to high
dimensionality (under one-hot encoding), so I will use Random Forest classifier.
This models work well in high dimensional spaces and deals with categorical inputs in quite a 
natural way.

## Training, validation and test sets

Because the task asks to provide a probabilistic model with a classification threshold that in a given 
month predicts 400 positive cases (as that is how many cases the fraud team can handle in a month)
and because we have 13 months of data (2017/01/01 - 2018/01/31), test_size will be set to 1/13 to have a reasonable performance estimate over 'the average month'. This will allow us to choose a classification threshold that produces up to 400 positive predictions and aligns
with the business use case. The instruction would be to present fraud analysts those samples which have a higher probability than the probability threshold selected.
