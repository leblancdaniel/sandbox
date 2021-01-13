import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


"""  IN PSEUDOCODE...

First, calculate a statistic for a bunch of samples w/ replacement...

    statistics = []
    for i in bootstraps:
        sample = select_sample_with_replacement(data)
        stat = calculate_statistic(sample)
        statistics.append(stat)

You can also boostrap model performance...

    statistics = []
    for i in bootstraps:
        train, test = select_sample_with_replacement(data, size)
        model = train_model(train)
        stat = evaluate_model(test)
        statistics.append(stat)

Then, calculate the confidence interval of your statistics...

    ordered_stats = sort(statistics)
    lower_bound = percentile(ordered_stats, (1-alpha)/2)
    upper_bound = percentile(ordered_stats, alpha+((1-alpha)/2))
"""

# load dataset
data = pd.read_csv('/Users/daniel/Documents/projects/pima-indians-diabetes-data.txt'
                    , sep=","
                    , header=None)
data.columns = ["Pregnancies", "Glucose", "BloodPressure"
                , "SkinThickness", "Insulin", "BMI"
                , "DiabetesPedigree", "Age", "Outcome"]
values = data.values 

# configure bootstrap
n_iterations = 1000
n_size = int(len(data) * 0.50)

# run bootstrap
stats = []
for i in range(n_iterations):
    # prepare train & test sets
    train = resample(values, n_samples=n_size)
    test = np.array([x for x in values if x.tolist() not in train.tolist()])
    # fit model
    model = DecisionTreeClassifier()
    model.fit(train[:, :-1], train[:, -1])
    # evaluate model
    predictions = model.predict(test[:, :-1])
    score = accuracy_score(test[:,-1], predictions)
    stats.append(score)

# plot scores on bootstrapped samples
plt.hist(stats)
plt.show()

alpha = 0.05
ntile = (alpha/2.0) * 100
lower_bound = max(0, np.percentile(stats, ntile))
ntile = ((1-alpha) + alpha/2.0) * 100
upper_bound = min(1, np.percentile(stats, ntile))

print("""
    {:.1f}% likelihood that the model 
    performance falls between {:.2f}% and {:.2f}%
    """.format((1-alpha)*100, lower_bound*100, upper_bound*100))

