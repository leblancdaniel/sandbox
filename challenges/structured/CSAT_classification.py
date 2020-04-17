import numpy as np
import pandas as pd
import pyodbc
from sqlalchemy import create_engine
#connect Python to SQL Tables
conn = pyodbc.connect('Driver={SQL Server};'
                        'Server=CHI000162199002\MSSQLSERVER02;'
                        'Database=master;'
                        'Trusted_Connection=yes;')
query = "SELECT * FROM master.dbo.[2019_CSAT_Surveys];"
df = pd.read_sql(query, conn)
#check null values and delete null columns
#print(df.isnull().mean()*100)
null_columns = ['CALL_REASON1', 'CALL_REASON2', 'NPS', 'IVR_LAST_STATE',
                'TRANSFER', 'CUSTOMER_ATTRITED', 'ESCALATED', 'REPEAT_CALL']
df = df.drop(null_columns, axis=1)
other_columns = ['BANK']
df = df.drop(other_columns, axis=1)
# Change data types
df['CALL_DATE'] = pd.to_datetime(df['CALL_DATE'])
df['CUSTOMER_TENURE'] = df['CUSTOMER_TENURE'].astype(float)
# standard imputation of null value
df['OUTSOURCED'] = df['OUTSOURCED'].fillna(df['OUTSOURCED'].mode()[0])
df['CALL_RESOLUTION'] = df['CALL_RESOLUTION'].fillna(0)
df['CALL_RESOLUTION'] = df['CALL_RESOLUTION'].astype(bool)
df['CUSTOMER_TENURE'] = df['CUSTOMER_TENURE'].fillna(df['CUSTOMER_TENURE'].mean())
# Create features
import datetime as dt
df['CALL_WEEKDAY'] = df['CALL_DATE'].dt.weekday_name
df['ACTIVE_DIGITAL'] = np.where(df['ACTIVE_ONLINE']==True, True,
                            np.where(df['ACTIVE_MOBILE']==True, True, False))
df['TALK_WRAP_TIME'] = df.apply(lambda row: row.HANDLE_TIME - row.HOLD_TIME, axis=1)
df['SATISFIED'] = np.where(df['CSAT']>8, True, False)
# re-read dataset
"""
print(df.isnull().mean()*100)
print(df.describe())
print(df.info())
"""
# matplotlib and seaborn for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
# showing histograms for all features
numeric_fields = ['CALL_HOUR', 'HANDLE_TIME', 'SPEED_TO_ANSWER', 'HOLD_TIME',
                    'CUSTOMER_TENURE', 'CSAT', 'AGENT_TENURE', 'TALK_WRAP_TIME']
"""
df[numeric_fields].hist(bins=25, color='red', edgecolor='black', linewidth=1,
            xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(1, 1, 1.2, 1.2))
# show heatmap to identify relationships between variables
corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()
"""
# transform non-normal distributions into normal distributions for heatmap
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
qt.fit_transform(df[numeric_fields].values.reshape(-1,1))
df['qt_CALL_HOUR'] = qt.fit_transform(df['CALL_HOUR'].values.reshape(-1,1))
df['qt_SPEED_TO_ANSWER'] = qt.fit_transform(df['SPEED_TO_ANSWER'].values.reshape(-1,1))
df['qt_HOLD_TIME'] = qt.fit_transform(df['HOLD_TIME'].values.reshape(-1,1))
df['qt_CUSTOMER_TENURE'] = qt.fit_transform(df['CUSTOMER_TENURE'].values.reshape(-1,1))
df['qt_AGENT_TENURE'] = qt.fit_transform(df['AGENT_TENURE'].values.reshape(-1,1))
df['qt_TALK_WRAP_TIME'] = qt.fit_transform(df['TALK_WRAP_TIME'].values.reshape(-1,1))
qt_num_fields = ['qt_CALL_HOUR', 'qt_HANDLE_TIME', 'qt_SPEED_TO_ANSWER',
                'qt_HOLD_TIME', 'qt_CUSTOMER_TENURE', 'qt_AGENT_TENURE',
                'qt_TALK_WRAP_TIME']
"""
df[qt_num_fields].hist(bins=25, color='red', edgecolor='black', linewidth=1,
            xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(1, 1, 1.2, 1.2))
plt.show()
"""
df = df.drop(numeric_fields, axis=1)
trash_variables = ['CALL_ID', 'CALL_DATE']
df = df.drop(trash_variables, axis=1)

# remove outliers (datapoints w/ Z-score of 3)
from scipy import stats
outlier_cols = ['qt_SPEED_TO_ANSWER', 'qt_HOLD_TIME', 'qt_TALK_WRAP_TIME']
print("Removing outliers of non-null values...")
df = df[(np.abs(stats.zscore(df[outlier_cols])) < 3).all(axis=1)]
print(df.shape)
df = df[[col for col in df.columns if col != 'SATISFIED'] + ['SATISFIED']]
print(df.info())

# encode columns with object type
from sklearn import preprocessing
from sklearn import utils
le = preprocessing.LabelEncoder()
for column in df.columns:
    if df[column].dtype == type(object):
        df[column] = le.fit_transform(df[column])

# identify dependent & independent variables
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#build train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
def run_model(model, alg_name):
    # fit model to training dataset
    model.fit(X_train, y_train)
    # make predictions for based on X test data
    y_pred = model.predict(X_test)
    # compare predictions with actual Y test data
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(alg_name + ": " + str(accuracy))
    # create and print confusion matrix
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    # create prediction probability based on applying model to test dataset
    y_pred_prob = model.predict_proba(X_test)[:,1]
    # calculate AUC score based on y_pred_prob and actual y from test dataset
    print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

from sklearn.ensemble import ExtraTreesClassifier
def n_best_features(X, y, n):
    # extract features from dataset, and derive importance based on entropy
    feat_extraction = ExtraTreesClassifier(n_estimators=100).fit(X, np.ravel(y))
    importances = feat_extraction.feature_importances_
    indicies = np.argsort(importances)[::-1]
    print("Top %d features: " % n)
    for i in range(n):
        print("%d. %s (%f)" % (i+1, X.columns.values[indicies[i]], importances[indicies[i]]))

#----- Decision Tree -----
from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
run_model(model, "Decision Tree")
n_best_features(X, y, 3)

#----- Random Forest -----
from sklearn.ensemble import RandomForestClassifier
y_train = np.ravel(y_train)

model = RandomForestClassifier(n_estimators=15, max_depth=5)
run_model(model, "Random Forest")
n_best_features(X, y, 3)

#----- xgBoost -----
from xgboost import XGBClassifier

model = XGBClassifier()
run_model(model, "XGBoost")
n_best_features(X, y, 3)

#----- Logistic Regression -----
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=5000)
run_model(model, "Logistic Regression")
n_best_features(X, y, 3)
"""
#----- SVM Classifier -----
from sklearn.svm import SVC

model = SVC(gamma='auto')
run_model(model, "SVM Classifier")
n_best_features(X, y, 3)
"""
#----- K-Nearest Neighbors -----
from sklearn import neighbors

model = neighbors.KNeighborsClassifier(n_neighbors=50)
run_model(model, "K-Nearest Neighbors")
n_best_features(X, y, 3)

#----- SGD Classifier -----
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

model = OneVsRestClassifier(SGDClassifier(max_iter=1000, tol=0, loss='log'))
run_model(model, "SGD Classifier")
n_best_features(X, y, 3)

#----- Gaussian Naive Bayes -----
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
run_model(model, "Gaussian Naive Bayes")
n_best_features(X, y, 3)

#----- Neural network - Multi-layer Perceptron -----
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(max_iter=1000, hidden_layer_sizes=(50,50,50))
run_model(model, "MLP Neural Network")
n_best_features(X, y, 3)
