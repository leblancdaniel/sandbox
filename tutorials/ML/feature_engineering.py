import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb
from sklearn import metrics 

# Load data
click_data = pd.read_csv('../input/feature-engineering-data/train_sample.csv',
                         parse_dates=['click_time'])
# Parse date feature
clicks = click_data.copy()
clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
clicks['second'] = clicks['click_time'].dt.second.astype('uint8')
# Create new columns in clicks using preprocessing.LabelEncoder()
cat_features = ['ip', 'app', 'device', 'os', 'channel']
encoder = preprocessing.LabelEncoder()
for feature in cat_features:
    encoded = encoder.fit_transform(clicks[feature])
    clicks[feature + '_labels'] = encoded
# Sort clicks
valid_fraction = 0.1
clicks_srt = clicks.sort_values('click_time')
valid_rows = int(len(clicks_srt) * valid_fraction)
# separate sorted clicks into training, validation, testing data
train = clicks_srt[:-valid_rows * 2]
valid = clicks_srt[-valid_rows * 2:-valid_rows]
test = clicks_srt[-valid_rows:]
# Define feature columns used to predict label 'is_attributed'
feature_cols = ['day', 'hour', 'minute', 'second', 
                'ip_labels', 'app_labels', 'device_labels',
                'os_labels', 'channel_labels']
# Create lgb Dataset objects to train, valid, test datasets
dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
dtest = lgb.Dataset(test[feature_cols], label=test['is_attributed'])
# Train baseline model
param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10)
# Evaluate model
ypred = bst.predict(test[feature_cols])
score = metrics.roc_auc_score(test['is_attributed'], ypred)
print(f"Test score: {score}")
