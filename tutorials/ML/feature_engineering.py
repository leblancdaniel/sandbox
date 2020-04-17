import os
import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb
from sklearn import metrics 
import category_encoders as ce

# Load data
script_dir = os.path.dirname(__file__)
rel_path = "train_sample.csv"
filepath = os.path.join(script_dir, rel_path)
click_data = pd.read_csv(filepath, parse_dates=['click_time'])

# Dataset split helper function
def get_data_splits(dataframe, valid_fraction=0.1):
    """ Splits a dataframe into train, validation, and test sets. First, orders by 
        the column 'click_time'. Set the size of the validation and test sets with
        the valid_fraction keyword argument.
    """
    # Parse date feature
    clicks = dataframe.copy()
    clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
    clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
    clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
    clicks['second'] = clicks['click_time'].dt.second.astype('uint8')
    clicks = clicks.sort_values('click_time')
    valid_rows = int(len(dataframe) * valid_fraction)
    train = clicks[:-valid_rows * 2]
    # valid size == test size, last two sections of the data
    valid = clicks[-valid_rows * 2:-valid_rows]
    test = clicks[-valid_rows:]
    
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
                    early_stopping_rounds=20, verbose_eval=False, 
                   boost_from_average=False)
    
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
train, valid, test = get_data_splits(click_data)
_ = train_model(train, valid)

print("CountEncoder model")
# Create new columns in clicks using encoder
cat_features = ['ip', 'app', 'device', 'os', 'channel']
train, valid, test = get_data_splits(click_data)
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
train, valid, test = get_data_splits(click_data)
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
train, valid, test = get_data_splits(click_data)
# Create the CatBoost encoder
cb_enc = ce.CatBoostEncoder(cols=cat_features)
# Learn encoding from the training set
cb_enc.fit(train[cat_features], train['is_attributed'])
# Apply encoding to the train and validation sets as new columns
train_encoded = train.join(cb_enc.transform(train[cat_features]).add_suffix('_cb'))
valid_encoded = valid.join(cb_enc.transform(valid[cat_features]).add_suffix('_cb'))
# Train model on CatBoost encoded datasets
_ = train_model(train_encoded, valid_encoded)
