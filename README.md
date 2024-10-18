# Heart Disease Prediction with Machine Learning

## Data Loading
   
I loaded the data, understood what the dataset looked like, and cleaned the data that contained missing or NaN values.
One consideration I made during this process included how I should handle missing values, and I ultimately decided to simply remove
them because the dataset was relatively clean. I considered using a KNNImputer to impute missing values, but this operation would be
expensive considering the size of the dataset and the choice of k. I also considered imputing with 0 but realized this would be non-representative
of the data.
  
## Exploratory Data Analysis

I visualized the target distribution and different risk factors as they related to heart disease. I created count plots using seaborn
to understand the prevalence of heart disease among different age groups in conjunction with a variety of risk factors. I visualized numerical data
using boxplots to check for outliers.

## Data Preprocessing

I split the data into train, validation, and test sets of 70%, 10%, and 20%, respectively. I encoded categorical attributes using ordinal encoding.
To handle the numerical data, I visualized various transformations such as logarithmic, nth-root, and yeo-johnson to achieve the most Gaussian-like 
transformation. Ultimately I used the yeo-johnson transformation for ['BMI', 'WeightInKilograms', 'HeightInMeters', 'SleepHours'] and the nth-root 
transformer for ['PhysicalHealthDays', 'MentalHealthDays']. Both the outputs were then scaled by a StandardScaler.

## Model Baselines

I used Logistic Regression as a baseline because the task was a binary classification. The baseline logistic regression model achieved an f1 score of 0.37.
The reason that I chose to use the f1-score as my main metric was because of the imbalance in the target distribution (roughly 94% of the data points did not have 
heart disease and 6% had heart disease). The f1-score will capture Type1 and Type2 errors as opposed to pure accuracy which could be misleading.

## Undersampling

Due to the imbalance in the target data distribution, I considered undersampling and oversampling. Undersampling algorithms remove data such that the target distribution
becomes more balanced and oversampling does the opposite. I chose undersampling because the dataset was already large (440,000 data points) to begin with. Ultimately, 
the undersampling technique RepeatedEditedNearestNeighbours achieved the greatest f1-score of 0.46. All of these undersampling strategies were testing using the same
logistic regression baseline model.

## Model Selection

I tested a variety of models including linear, SVMs, tree-based, ensembling, boosting, and neural networks. Ultimately, based on a variety of metrics (f1-score, auc-score,
recall, precision, and accuracy) LGBMClassifier achieved the best performance.

## Hyperparameter Tuning with Cross Validation

Lastly, I tuned my model using an exhaustive grid search and found the optimal parameters for the LGBM model. This result was verified with
a cross-validation split of 5. The final model achieved an accuracy of 0.94 and and f1-score of 0.47.
   
