import pandas as pd 
import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# simulate data samples
np.random.seed(42)
sample1 = sorted(list(np.random.randn(1000)))
sample2 = sorted(list(2 * np.random.randn(1000)))
print("sample1 mean={:.2f} stdev={:.2f}".format(np.mean(sample1), np.std(sample1)))
print("sample2 mean={:.2f} stdev={:.2f}".format(np.mean(sample2), np.std(sample2)))


# Import data and describe data
data = pd.read_csv("/Users/daniel/Documents/projects/500Person_GHWI.csv")
print(data.head())
print(data.info()) 
print(data.describe()) 

# print count of unique values and unique values of column 'Gender'
print(data['Gender'].nunique())
print(data['Gender'].unique())

# create histograms of each variable

plt.style.use('ggplot')

data['Height'].plot(kind='hist')
plt.title('Distribution of Height')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.show()

data['Weight'].plot(kind='hist')
plt.title('Distribution of Weight')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()

data['Index'].plot(kind='hist')
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

# create histograms of each variable broken out by gender
data[data['Gender'] == 'Male'].Height.plot(kind='hist', color='blue', alpha=0.5)
data[data['Gender'] == 'Female'].Height.plot(kind='hist', color='magenta', alpha=0.5)
plt.legend(labels=['Males', 'Females'])
plt.title('Dist. of Height by Gender')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.show()

data[data['Gender'] == 'Male'].Weight.plot(kind='hist', color='blue', alpha=0.5)
data[data['Gender'] == 'Female'].Weight.plot(kind='hist', color='magenta', alpha=0.5)
plt.legend(labels=['Males', 'Females'])
plt.title('Dist. of Weight by Gender')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()

data[data['Gender'] == 'Male'].Index.plot(kind='hist', color='blue', alpha=0.5)
data[data['Gender'] == 'Female'].Index.plot(kind='hist', color='magenta', alpha=0.5)
plt.legend(labels=['Males', 'Females'])
plt.title('Dist. of BMI by Gender')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

# scipy.stats.describe also gives skewness and kurtosis :)
print(scs.describe(data[['Height', 'Weight', 'Index']]))

male_data = data[data['Gender'] == 'Male']
print(scs.describe(male_data[['Height', 'Weight', 'Index']]))

female_data = data[data['Gender'] == 'Female']
print(scs.describe(female_data[['Height', 'Weight', 'Index']]))


# create scatter plot to show relationship between height and weight
ax1 = data[data['Gender'] == 'Male'].plot(kind='scatter', x='Height', y='Weight', color='blue', alpha=0.5)
data[data['Gender'] == 'Female'].plot(kind='scatter', x='Height', y='Weight', color='magenta', alpha=0.5, ax=ax1)
plt.legend(labels=['Males', 'Females'])
plt.title('Relationship between Height & Weight')
plt.xlabel('Height')
plt.ylabel('Weight')


# obtain best fit line with numpy.polyfit(x, y, degree)
male_fit = np.polyfit(male_data['Height'], male_data['Weight'], 1)
print(male_fit)
female_fit = np.polyfit(female_data['Height'], female_data['Weight'], 1)
print(female_fit)

# add regression lines to scatterplot
plt.plot(male_data['Height'], male_fit[0] * male_data['Height'] + male_fit[1], color='darkblue', linewidth=2)
plt.plot(female_data['Height'], female_fit[0] * female_data['Height'] + female_fit[1], color='deeppink', linewidth=2)
# add regression equations
plt.text(150, 136, 'y={:.2f}+{:.2f}*x'.format(male_fit[1], male_fit[0]), color='darkblue', size=12)
plt.text(184, 80, 'y={:.2f}+{:.2f}*x'.format(female_fit[1], female_fit[0]), color='deeppink', size=12)

plt.show()


# create regression plot with seaborn
fig = plt.figure()
sns.regplot(x=male_data['Height'], y=male_data['Weight'], color='blue')
sns.regplot(x=female_data['Height'], y=female_data['Weight'], color='magenta')
plt.legend(labels=['Males', 'Females'])
plt.title('Relationship between Height and Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

# linear regression model w/ sklearn
lr_males = LinearRegression()
lr_males.fit(male_data[['Height']], male_data['Weight'])
print(lr_males.intercept_, lr_males.coef_)

lr_females = LinearRegression()
lr_females.fit(female_data[['Height']], female_data['Weight'])
print(lr_females.intercept_, lr_females.coef_)

# predictions using numpy.polyval
female_fit = np.polyfit(female_data['Height'], female_data['Weight'], 1)
print(np.polyval(female_fit, [150]))
# predictions using sklearn
print(lr_females.predict([[150]]))


# Pearson correlation coefficient (built-in)
print(male_data.corr())
print(female_data.corr())
# Pearson coefficient w/ scipy.stats
pearson_coef, p_val = scs.pearsonr(female_data['Index'], female_data['Weight'])
print(pearson_coef, p_val)


# Residual plots
fig = plt.figure()
sns.residplot(female_data['Weight'], female_data['Index'], color='magenta')
plt.title('Residual plot Weight vs BMI')
plt.xlabel('Weight')
plt.ylabel('Index')
plt.show()


# multiple linear regression
# turn categorical variable into dummy variables and drop one, to avoid multi-collinearity
dummy_data = pd.get_dummies(data)
dummy_data.drop('Gender_Female', axis=1, inplace=True)
dummy_data.rename(columns={'Gender_Male': 'Gender'}, inplace=True)
print(dummy_data.head())

mlr = LinearRegression()
# reads: fit a model that uses height and gender to predict weight
mlr.fit(dummy_data[['Height', 'Gender']], dummy_data['Weight'])
print(mlr.intercept_, mlr.coef_)
print("Weight = {:.2f} + {:.2f}*Height + {:.2f}*Gender".format(mlr.intercept_, 
