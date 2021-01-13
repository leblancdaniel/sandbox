import pandas as pd 
import numpy as np 

# load data
data = pd.read_csv('/Users/daniel/Documents/projects/product-sample-data.csv')
print(data.head()) 
print(data.info())

# Step 1: define goals of test, hypothesis
"""
Determine which banner produces more sales, defined in 'Target' column

IF the banner has something on it or designed differently,
THEN we will see an increase in sales (usually the effect size/ or minimum detectable effect given by expert),
BECAUSE the process will be better for the customer
"""

# Step 2: determine sample method of test
"""
Random sample of website visitors
Be cognizant of network effects and other factors that could skew results
(new vs. old users, demographics, different countries?)
"""

# Step 3: determine appropriate significance level (alpha)
"""
0.05, unless we have strong reason to believe that a false positive would be very costly
"""

# Step 4: Conduct power analysis to determine sample size
"""
Use effect size, power level, and alpha to determine minimum sample size needed

Usually expert will give you a sense of minimum detectable effect.
Power level should be 80-90% typically.  
    Power is the inverse of Beta (False Negative level).  
    AKA the % chance of correctly detecting an effect
You already determined alpha
"""
import scipy.stats as scs

def get_sample_size(mean, effect_size, power=0.90, alpha=0.05):
    
    standard_norm = scs.norm(0,1)
    Z_alpha = standard_norm.ppf(1-alpha/2)
    Z_beta = standard_norm.ppf(power)

    p = (mean + (mean + effect_size)) / 2

    min_N = (2 * p * (1-p) * (Z_alpha + Z_beta)**2 / effect_size**2)

    print("""
        With a significance level of {}%,
        you need a minimum sample size of {:.2f} per group
        to have a {}% chance of correctly detecting
        an difference in the metric of {}.
        """.format((1-alpha)*100, min_N, power*100, effect_size))

    return min_N

get_sample_size(mean=0.05, effect_size=0.05, power=0.80, alpha=0.01)
get_sample_size(mean=0.05, effect_size=0.05, power=0.80, alpha=0.05)
get_sample_size(mean=0.05, effect_size=0.05, power=0.90, alpha=0.05)
get_sample_size(mean=0.05, effect_size=0.01, power=0.90, alpha=0.05)

# Step 5: Run test and collect samples
"""
This is the data you'll be working with!
You'll know when to stop the test when you reach the minimum sample size,  
you could also determine how long that takes if you know the daily traffic of your website

However, be cognizant of network effects.  
You could determine those by running another test in parallel which stratifies users
  into small clusters with similar characteristics, then separated in half - 
  randomized at the cluster and individual levels (half of each group in control/test group),
  then results of the AB test are generated for all.  Hypothesis test will test if there is a difference
  between the individual and cluster group means.  
  If there isn't any interference, then the results should be roughly the same.
"""

# Step 6: Analyze results across samples
n_by_product = data.groupby("product")["target"].mean()
print(n_by_product)

df_accessories = data[data["product"] == 'accessories']
df_clothes = data[data["product"] == 'clothes']
df_company = data[data["product"] == 'company']
df_sneakers = data[data["product"] == 'sneakers']
df_sports_nutrition = data[data["product"] == 'sports_nutrition']

# t-test tests difference of mean between two independent samples (numerical)
# chi-sq tests difference between categorical values
print("P-values indicate the likelihood of getting this test result if the null hyp. were true")
t_stat, p_val = scs.ttest_ind(df_accessories['target'], df_clothes['target'])
print(t_stat, p_val)
t_stat, p_val = scs.ttest_ind(df_accessories['target'], df_company['target'])
print(t_stat, p_val)
t_stat, p_val = scs.ttest_ind(df_accessories['target'], df_sneakers['target'])
print(t_stat, p_val)
t_stat, p_val = scs.ttest_ind(df_accessories['target'], df_sports_nutrition['target'])
print(t_stat, p_val)

# Step 6 (alternative): Bootstrap sample results
from sklearn.utils import resample
import matplotlib.pyplot as plt

def bootstrap(df, n_bootstraps=1000, alpha=0.05):

    statistics = []
    for _ in range(n_bootstraps):
        sample = resample(df['target'].values, n_samples=int(len(df)))
        stat = np.mean(sample)
        statistics.append(stat)

    ordered_stats = sorted(statistics)
    ntile = alpha/2 * 100
    lower_bound = max(0, np.percentile(ordered_stats, ntile))
    ntile = ((1-alpha) + (alpha/2)) * 100
    upper_bound = min(1, np.percentile(ordered_stats, ntile))

    print("""
        Mean: {}
        There is a {}% chance that the mean
        falls in between {:.4f} and {:.4f}.
        """.format(np.mean(ordered_stats), (1-alpha)*100, lower_bound, upper_bound))

bootstrap(df_accessories)
bootstrap(df_clothes)
bootstrap(df_company)
bootstrap(df_sneakers)
bootstrap(df_sports_nutrition)
