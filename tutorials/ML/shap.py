import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
from matplotlib import pyplot as plt
import shap

# Load data
script_dir = os.path.dirname(__file__)
rel_path = "hospital_readmissions.csv"
filepath = os.path.join(script_dir, rel_path)
data = pd.read_csv(filepath)
y = data.readmitted
base_features = [c for c in data.columns if c != "readmitted"]
X = data[base_features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
#eli5.show_weights(perm, feature_names=val_X.columns.tolist())
print(eli5.format_as_text(eli5.explain_weights(perm)))

# Calculate SHAP values
data_for_prediction = val_X.iloc[0,:]  # use 1 row of data here. Could use multiple rows if desired
# Create object that can calculate shap values
explainer = shap.KernelExplainer(my_model)
shap_values = explainer.shap_values(val_X)
shap.summary_plot(shap_values[1], val_X)

for col in val_X.columns:
    shap.dependence_plot(col, shap_values[1], val_X)
