import pandas as pd
from sklearn import preprocessing, metrics
import lightgbm as lgb
import category_encoders as ce
import itertools

# Load data
script_dir = os.path.dirname(__file__)
rel_path = "train_sample.csv"
filepath = os.path.join(script_dir, rel_path)
click_data = pd.read_csv(filepath, parse_dates=['click_time'])
click_times = click_data['click_time']
clicks = click_data.assign(day=click_times.dt.day.astype('uint8'),
                           hour=click_times.dt.hour.astype('uint8'),
                           minute=click_times.dt.minute.astype('uint8'),
                           second=click_times.dt.second.astype('uint8'))

# Label encoding for categorical features
cat_features = ['ip', 'app', 'device', 'os', 'channel']
for feature in cat_features:
    label_encoder = preprocessing.LabelEncoder()
    clicks[feature] = label_encoder.fit_transform(clicks[feature])

# Dataset split helper function
def get_data_splits(dataframe, valid_fraction=0.1):
    """ Splits a dataframe into train, validation, and test sets. First, orders by 
        the column 'click_time'. Set the size of the validation and test sets with
        the valid_fraction keyword argument.
    """
    dataframe = dataframe.sort_values('click_time')
    valid_rows = int(len(dataframe) * valid_fraction)
    train = dataframe[:-valid_rows * 2]
    valid = dataframe[-valid_rows * 2:-valid_rows]
    test = dataframe[-valid_rows:]
    
    return train, valid, test

# LGBM Train + Eval helper functions
def train_model(train, valid, test=None, feature_cols=None):
    if feature_cols is None:
        feature_cols = train.columns.drop(['click_time', 'attributed_time',
                                           'is_attributed'])
    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
    
    param = {'num_leaves': 64, 'objective': 'binary', 
             'metric': 'auc', 'seed': 7}
    num_round = 1000
    print("Training model!")
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], 
                    early_stopping_rounds=20, verbose_eval=False)
    
    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)
    print(f"Validation AUC score: {valid_score}")
    
    if test is not None: 
        test_pred = bst.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)
        return bst, valid_score, test_score
    else:
        return bst, valid_score

print("Baseline model")
train, valid, test = get_data_splits(clicks)
_ = train_model(train, valid)
"""
print("CountEncoder model")
# Create new columns in clicks using encoder
cat_features = ['ip', 'app', 'device', 'os', 'channel']
train, valid, test = get_data_splits(clicks)
# create and train CountEncoder on categorcal columns
count_enc = ce.CountEncoder(cols=cat_features)
count_enc.fit(train[cat_features])
# Apply encoding to the train and validation sets as new columns
train_encoded = train.join(count_enc.transform(train[cat_features]).add_suffix('_count'))
valid_encoded = valid.join(count_enc.transform(valid[cat_features]).add_suffix('_count'))
# Train the model on the count encoded datasets
_ = train_model(train_encoded, valid_encoded)

print("TargetEncoder model")
# Remove 'ip' column to improve TargetEncoder and CatBoostEncoder results
cat_features = ['app', 'device', 'os', 'channel']
train, valid, test = get_data_splits(clicks)
# Create the target encoder. You can find this easily by using tab completion.
target_enc = ce.TargetEncoder(cols=cat_features)
target_enc.fit(train[cat_features], train['is_attributed'])
# Apply encoding to the train and validation sets as new columns
train_encoded = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))
valid_encoded = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))
# Train model on target encoded datasets
_ = train_model(train_encoded, valid_encoded)

print("CatBoostEncoder model")
# Do the same for CatBoostEncoder
train, valid, test = get_data_splits(clicks)
# Create the CatBoost encoder
cb_enc = ce.CatBoostEncoder(cols=cat_features)
# Learn encoding from the training set
cb_enc.fit(train[cat_features], train['is_attributed'])
# Apply encoding to the train and validation sets as new columns
train_encoded = train.join(cb_enc.transform(train[cat_features]).add_suffix('_cb'))
valid_encoded = valid.join(cb_enc.transform(valid[cat_features]).add_suffix('_cb'))
# Train model on CatBoost encoded datasets
_ = train_model(train_encoded, valid_encoded)
"""
# Feature generation based on interactions
cat_features = ['ip', 'app', 'device', 'os', 'channel']
interactions = pd.DataFrame(index=clicks.index)
# Iterate through each pair of features, combine them into interaction features
for c1, c2 in itertools.combinations(cat_features, 2):
    new_col_name = '_'.join([c1, c2])
    new_values = clicks[c1].map(str) + "_" + clicks[c2].map(str)
    encoder = preprocessing.LabelEncoder()
    interactions[new_col_name] = encoder.fit_transform(new_values)
# combine interaction features with clicks dataset
clicks = clicks.join(interactions)

# Generate numerical features based on rolling window
# Number of events in the past X hours
def count_past_events(series, window='6H'):
    """ Returns a series that counts the number of events in the past 6 hours """
    series = pd.Series(series.index, index=series)
    past_events = series.rolling(window).count() - 1
    return past_events
clicks['ip_past_6H_counts'] = count_past_events(clicks['ip'], '6H')
# Time since last event
def time_diff(series):
    """ Returns a series with the time since the last timestamp in seconds """
    return series.diff().dt.total_seconds()
timedeltas = clicks.groupby('ip')['click_time'].transform(time_diff)
clicks['past_events_6H'] = timedeltas
# Number of previous app downloads
def previous_attributions(series):
    """ Returns a series with the rolling sum of target series since current row """
    sums = series.expanding(min_periods=2).sum() - series
    return sums
clicks['ip_past_6hr_counts'] = previous_attributions(clicks['is_attributed'])
# split and train model on new features: interactions, past counts, time since last, rolling sum
train, valid, test = get_data_splits(clicks.join(timedeltas))
_ = train_model(train, valid, test)
