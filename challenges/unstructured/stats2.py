# import libraries for data reading, cleansing, processing, visualizations
import os
import numpy as np
import pandas as pd
from scipy.stats import describe
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(4)

script_dir = os.path.dirname(__file__)
rel_path = "ds2.csv"
filepath = os.path.join(script_dir, rel_path)
data = pd.read_csv(filepath)
data.columns = ['ID', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
data = data.drop(columns='ID', axis=1)
print(round(data.head(),2))
print(data.info())

for n in data.columns:
    axis = 0
    desc = describe(data[n], axis=0)
    print(n, ": \n", desc, "\n")
    axis += 1

sns.set(color_codes=True)
# violin plots to show ranges and distributions for all features
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.violinplot(data=data, ax=ax)
plt.show()
# show heatmap to identify relationships between variables
corr = data.corr()
sns.heatmap(round(corr,2), annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()
# visualize relationships between variables with a pairplot
sns.pairplot(data, height=1, aspect=1,
                plot_kws=dict(s=1, edgecolor='b'), diag_kind="kde")
plt.show()

# identify 3 clusters using x1 and x6 (bimodal features)
from sklearn.cluster import KMeans
def kmeans(X, nclust=3):
    model = KMeans(nclust).fit(X)
    clust_labels = model.predict(X)
    center = model.cluster_centers_
    return (clust_labels, center)

clust_labels, center = kmeans(data, 3)
km = pd.DataFrame(clust_labels)
print(km.head(10))
data.insert((data.shape[1]), 'km', km)

plt.scatter(data['x1'], data['x6'], c=km[0], s=1)
plt.show()
