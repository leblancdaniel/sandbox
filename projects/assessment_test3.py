import pandas as pd 
import numpy as np 

# load data
data = pd.read_csv('/Users/daniel/Documents/projects/homepage_actions.csv')
print(data.head())
print(data.describe()) 
print(data["group"].unique())

# step 1: define goal, metric
"""
Increase clickthrough rate
"""

# step 2: define signicance level, min detectable effect, power level
"""
Do we have a baseline?  If not, should consider an A/A test.

alpha = 0.05
mde = 10% growth on CTR metric
power = 90%
"""

# step 3: power analysis
import scipy.stats as scs

def power_analysis(mean, mde, power=0.90, alpha=0.05):
    stats_norm = scs.norm(0,1)
    Z_alpha = stats_norm.ppf(1-alpha/2)
    Z_beta = stats_norm.ppf(power)
    p = (mean + mean + mde) / 2
    print(Z_alpha, Z_beta)

    min_n = (2 * p * (1-p) * (Z_beta + Z_alpha)**2) / mde**2

    print("""
        With a confidence level of {}%
        we need a minimum sample size of {} per group 
        in order to correctly detect an effect size of {}
        at {}% of the time.
        """.format((1-alpha)*100, min_n, mde, power*100))

power_analysis(0.2185, 0.02)
power_analysis(0.2185, 0.02, alpha=0.01)

# step 4: define sampling method and run test
"""
randomly select website visitors
min_n = 10000 per group
daily traffic = 2000
would take about 10 days to collect sample

collect data
"""

# step 5: analyze data

data["action"] = np.where(data["action"] == 'click', 1, 0)
print(data)
print(data["action"].unique())

df = data.groupby("group")["action"].mean()
print(df)

ctrl_group = data[data["group"] == 'control']
exp_group = data[data["group"] == 'experiment']

tstat, pval = scs.ttest_ind(ctrl_group["action"], exp_group["action"])
print("T-statistic:", tstat)
print("p-value:", pval)

# Step 5 (alternative): bootstrap
from sklearn.utils import resample

def bootstrap(df, n_bootstraps=10000, alpha=0.05):
    statistics = []
    for _ in range(n_bootstraps):
        sample = resample(df["action"].values, n_samples=int(len(df)))
        stat = np.mean(sample)
        statistics.append(stat)
    ordered_stats = sorted(statistics)
    ntile = alpha/2 * 100
    lower_bound = max(0, np.percentile(ordered_stats, ntile))
    ntile = ((1-alpha) + (alpha/2)) * 100
    upper_bound = min(1, np.percentile(ordered_stats, ntile))

    print("Lower:", lower_bound, "Mean:", np.mean(ordered_stats), "Upper:", upper_bound)

bootstrap(ctrl_group)
bootstrap(exp_group)

def bootstrap_list(df, n_bootstraps=10000, alpha=0.05):
    statistics = []
    for _ in range(n_bootstraps):
        sample = resample(df["action"].values, n_samples=int(len(df)))
        stat = np.mean(sample)
        statistics.append(stat)
    ordered_stats = sorted(statistics)
    return ordered_stats

tstat, pval = scs.ttest_ind(bootstrap_list(ctrl_group), bootstrap_list(exp_group))
print(tstat, pval)