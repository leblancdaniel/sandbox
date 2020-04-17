# numpy, pandas, stats for data cleansing, processing
import os
import numpy as np
import pandas as pd
import scipy
from scipy import stats
# matplotlib and seaborn for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
# for splitting data into train/test datasets for Random Forest Classifier model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# LIME explainability model for tabular datasets
import lime
import lime.lime_tabular
# import researchpy for descriptive statistics and informative t-test results
import researchpy as rp
np.random.seed(4)

# read diabetes dataset
script_dir = os.path.dirname(__file__)
rel_path = "diabetes.csv"
filepath = os.path.join(script_dir, rel_path)
data = pd.read_csv(filepath, engine='python')
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
print(data.head())
print(data.describe())
print(data.info())

# flag zero-values for low-quality test results
vital_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[vital_columns] = data[vital_columns].replace(0, np.NaN)
# print proportion of NULL values within each vital feature
print(data.isnull().mean()*100)

# impute median in NULL values for features with <5% Null values
data['Glucose'] = data['Glucose'].fillna(data['Glucose'].median())
data['BloodPressure'] = data['BloodPressure'].fillna(data['BloodPressure'].median())
data['BMI'] = data['BMI'].fillna(data['BMI'].median())

#identify R-squared between features with large % of missing values
corr = data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

# SkinThickness has strongest relationship with BMI
# impute mean value for SkinThickness by BMI group (discrete buckets)
data['BMI_Group'] = pd.cut(data['BMI'], 10)
data['SkinThickness'] = data.groupby('BMI_Group').transform(lambda x: x.fillna(x.mean()))
# Insulin has strongest relationship with Glucose
# impute value outside normal range for missing Insulin values
data['Glucose_Group'] = pd.cut(data['Glucose'], 10)
data['Insulin'] = data.groupby('Glucose_Group').transform(lambda x: x.fillna(x.mean()))
# drop newly created columns
data = data.drop(columns=['BMI_Group', 'Glucose_Group'])
# re-read dataset
print(data.describe())
print(data.info())
# remove outliers (datapoints w/ Z-score of 3)
print("Removing outliers of non-null values...")
data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
print(data.shape)

# identify dependent & independent variables
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# transform X and y
X = X.astype(float)
le = sklearn.preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
class_names = le.classes_

# separate dataset into 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
# train random forest classifier
rf = RandomForestClassifier(n_estimators=500).fit(X_train, y_train)
# make predictions for test dataset
y_pred = rf.predict(X_test)
# calculate accuracy score
print(accuracy_score(y_test, y_pred) * 100)

# create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
            feature_names=features,
            class_names=class_names,
            discretize_continuous=True)
# i is the record we explain
i = 0
exp = explainer.explain_instance(X_test.values[i], rf.predict_proba, num_features=8)
exp.show_in_notebook(show_table=True, show_all=True)

# separate postive and negative outcomes for independent t-test
positive_outcome = data[data['Outcome'] == 1]
negative_outcome = data[data['Outcome'] == 0]
# rename 'Glucose' field to clarify test results
positive_outcome = positive_outcome.rename({'Glucose': 'Positive_Glucose'}, axis=1)
negative_outcome = negative_outcome.rename({'Glucose': 'Negative_Glucose'}, axis=1)
# run t-test to determine difference of means in glucose for positive/negative outcomes
descriptive_stats, test_results = rp.ttest(
                                    negative_outcome['Negative_Glucose'],
                                    positive_outcome['Positive_Glucose'])
print(descriptive_stats)
print(test_results)
