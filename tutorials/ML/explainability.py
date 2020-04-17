# Loading data, dividing, modeling and EDA below
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import eli5
from eli5.sklearn import PermutationImportance

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Load data
script_dir = os.path.dirname(__file__)
rel_path = "train.csv"
filepath = os.path.join(script_dir, rel_path)
data = pd.read_csv(filepath)

# Remove data with extreme outlier coordinates or negative fares
data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' + 
                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +
                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +
                  'fare_amount > 0'
                  )
y = data.fare_amount
base_features = ['pickup_longitude',
                 'pickup_latitude',
                 'dropoff_longitude',
                 'dropoff_latitude',
                 'passenger_count']
X = data[base_features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=50, random_state=1).fit(train_X, train_y)

# show permutation importance 
perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)
#eli5.show_weights(perm, feature_names = val_X.columns.tolist())
print(eli5.format_as_text(eli5.explain_weights(perm)))

# create new features
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)
features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']
X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y
perm2 = PermutationImportance(second_model, random_state=1).fit(new_val_X, new_val_y)
#eli5.show_weights(perm2, feature_names = features_2)
print(eli5.format_as_text(eli5.explain_weights(perm2)))

# Partial Dependence Plot
for feat_name in base_features:
    pdp_dist = pdp.pdp_isolate(model=first_model, dataset=new_val_X, 
                               model_features=features_2, feature=feat_name)
    pdp.pdp_plot(pdp_dist, feat_name)
    plt.show()

# 2D partial dependence plot
inter2 = pdp.pdp_interact(model=first_model, dataset=new_val_X, model_features=features_2, features=['pickup_longitude', 'dropoff_longitude'])
pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['pickup_longitude', 'dropoff_longitude'], plot_type='contour')
plt.show()
