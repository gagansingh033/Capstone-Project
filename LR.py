# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 09:24:00 2021

@author: Gagan Saini
"""
# Importing Requried libraries
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

class Input_read():
    def __init__(self, input_file_dir, input_file_name):
        self.input_file_dir = input_file_dir
        self.input_file_name = input_file_name


# Input csv file
input_file_dir = Path(r'C:\Users\Gagan Saini\Desktop\Study\sem_4\cap_project')
input_file_name = 'acs2017_census_tract_data.csv'

read_file = Input_read(input_file_dir, input_file_name)

df =  pd.read_csv(read_file.input_file_dir / read_file.input_file_name)

# First 5 rows of the dataset
df.head(5)

# dropping the columns which is not required
df = df.drop(['IncomeErr', 'IncomePerCapErr'], 1)

# Rename the columns
df.rename(columns={'TotalPop': 'TotalPopulation'}, inplace=True)

# Checking the column name is changed
df.columns.values

# Change percentage
def percent(expression="1000*12%") -> float:
    if "%" in expression:
        expression = expression.replace("%","")
        a, b = expression.split("*")
        a, b = float(a), float(b)
    return a*b/100


df['Hisp_in_num'] = [percent("{}*{}%".format(x,y)) for x, y in zip(df['TotalPopulation'], df['Hispanic'])]
df['White_in_num'] = [percent("{}*{}%".format(x,y)) for x, y in zip(df['TotalPopulation'], df['White'])]
df['Black_in_num'] = [percent("{}*{}%".format(x,y)) for x, y in zip(df['TotalPopulation'], df['Black'])]
df['Native_in_num'] = [percent("{}*{}%".format(x,y)) for x, y in zip(df['TotalPopulation'], df['Native'])]
df['Asian_in_num'] = [percent("{}*{}%".format(x,y)) for x, y in zip(df['TotalPopulation'], df['Asian'])]
df['Pacific_in_num'] = [percent("{}*{}%".format(x,y)) for x, y in zip(df['TotalPopulation'], df['Pacific'])]


# chceking shape of dataframe
df.shape

# Descibing the dataset with stat
df.describe()

print('Ploting Scatter, bar and Heatmap graphs for Race')
# plots about Race
fig, axes = plt.subplots(nrows=2, ncols=3)
fig.set_figheight(5)
fig.set_figwidth(20)

# all Race columns check
df.plot(ax=axes[0,0],x='TotalPopulation', y='Hisp_in_num', kind = 'scatter', title = 'TotalPop vs. Hispanic Density')
df.plot(ax=axes[0,1],x='TotalPopulation', y='White_in_num', kind = 'scatter', title = 'TotalPop vs. White Density')
df.plot(ax=axes[0,2],x='TotalPopulation', y='Black_in_num', kind = 'scatter', title = 'TotalPop vs. Black Density')
df.plot(ax=axes[1,0],x='TotalPopulation', y='Native_in_num', kind = 'scatter', title = 'TotalPop vs. Native Density')
df.plot(ax=axes[1,1],x='TotalPopulation', y='Asian_in_num', kind = 'scatter', title = 'TotalPop vs. Asian Density')
df.plot(ax=axes[1,2],x='TotalPopulation', y='Pacific_in_num', kind = 'scatter', title = 'TotalPop vs. Pacific Density')

plt.tight_layout()
plt.close()

# Race is in the US as a percentage of the total 
race_columns = ['Hispanic','White','Black','Native','Asian','Pacific']
census_races =df[race_columns]
(census_races.sum()/len(census_races)).plot.bar(title = "Percentage of Americans by Race")

# Heatmap with corr
f = (df.loc[:, ['Hispanic', 'Black', 'White', 'Asian', 'Native','Pacific']]).corr()
sns.heatmap(f, annot=True)


#########################################################################
print('Ploting Scatter, bar and Heatmap graphs for Poverty')
# plots about Poverty with professions
fig, axes = plt.subplots(nrows=2, ncols=3)
fig.set_figheight(5)
fig.set_figwidth(20)

# all Poverty columns check
df.plot(ax=axes[0,0],x='Professional', y='Poverty', kind = 'scatter', title = 'TotalPopulation vs. Poverty')
df.plot(ax=axes[0,1],x='Service', y='Poverty', kind = 'scatter', title = 'Service  vs. Poverty')
df.plot(ax=axes[0,2], x='Office', y='Poverty', kind = 'scatter', title = 'Office vs. Poverty')
df.plot(ax=axes[1,0], x='Construction', y='Poverty', kind = 'scatter', title = 'Construction vs. Poverty')
df.plot(ax=axes[1,1], x='Production', y='Poverty', kind = 'scatter', title = 'Production vs. Poverty')
df.plot(ax=axes[1,2], x='WorkAtHome', y='Poverty', kind = 'scatter', title = 'WorkAtHome vs. Poverty')


plt.tight_layout()
plt.close()

# Poverty is in the US as a percentage of the total 
poverty_columns = ['Professional','Service','Office','Construction','Production','WorkAtHome']
poverty_columns =df[poverty_columns]
(poverty_columns.sum()/len(poverty_columns)).plot.bar(title = "Percentage of Americans by Poverty")

# Heatmap with corr
f = (df.loc[:, ['Professional','Service','Office','Construction','Production','WorkAtHome']]).corr()
sns.heatmap(f, annot=True)


#########################################################################
print('Ploting Scatter, bar and Heatmap graphs for Income')
# plots about income with Men and Women
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(5)
fig.set_figwidth(20)

# all income columns check
df.plot(ax=axes[0,0], x='Income', y='TotalPopulation', kind = 'scatter', title = 'Income vs. TotalPopulation')
df.plot(ax=axes[0,1], x='Income', y='Women', kind = 'scatter', title = 'Income vs. Women')
df.plot(ax=axes[1,0], x='Income', y='Men', kind = 'scatter', title = 'Income vs. Men')
df.plot(ax=axes[1,1], x='Income', y='VotingAgeCitizen', kind = 'scatter', title = 'Income vs. VotingAgeCitizen')

plt.tight_layout()
plt.show()

# income is in the US as a percentage of the total 
income_columns = ['TotalPopulation', 'Men','Women', 'VotingAgeCitizen']
income_columns =df[income_columns]
(income_columns.sum()/len(income_columns)).plot.bar(title = "Percentage of Americans by Income")

# Heatmap with corr
f = (df.loc[:, ['TotalPopulation', 'Men','Women', 'VotingAgeCitizen']]).corr()
sns.heatmap(f, annot=True)

df.to_csv(input_file_dir / "dataframe for heatmaps.csv")
##############################################################################

df_County = df.groupby('County').sum()

df_county = pd.DataFrame()
for i in df["County"].unique():
    counties_df = df[df['County'] == i]
    counties_df.loc[df.index[-1] + 1] = counties_df.sum()
    counties_df['State'] = list(counties_df["State"][:1])[0]
    counties_df['County'] = list(counties_df["County"][:1])[0]
    df_county = df_county.append(counties_df[-1:])

df_selected = df[['State', 'County', 'TotalPopulation', 'Men', 'Women',
       'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific','Hisp_in_num', 'White_in_num',
       'Black_in_num', 'Native_in_num', 'Asian_in_num', 'Pacific_in_num']]

# L R

#################################### Best fit line ###########
print('Ploting graph for Best fit line')
y = df_county['TotalPopulation'].reset_index(drop=True)
#x1 = df['Men']
x1 = df_county['Women'].reset_index(drop=True)

# Plot
plt.scatter(x1,y)
plt.xlabel(x1.name, fontsize = 20)
plt.ylabel(y.name, fontsize = 20)
plt.show()

# Adding constants
x = sm.add_constant(x1)
 
results = sm.OLS(y,x).fit()

print('Output of Ordinary Least-Squares (OLS) Regression')
print(results.summary())



### Ploting Regression line

print('Ploting Regression line on the basis of OLS output')
plt.scatter(x1,y)
 
y_line = 1.9650*x1 + (324.4823)
 
fig = plt.plot(x1,y_line, lw=2, c="orange", label = "regression line")
 
plt.xlabel(x1.name, fontsize = 20)
 
plt.ylabel(y.name, fontsize = 20)
 
plt.show()


########## train and test

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(x1,y,test_size=0.3,random_state=23)
linreg = LinearRegression()
linreg.fit(X_train.values.reshape(-1, 1),y_train.values.reshape(-1, 1))


#Intercept and Coefficient
print("Intercept: ", linreg.intercept_)
print("Coefficient: ", linreg.coef_)

#Prediction of test set
y_pred_lr= linreg.predict(X_test.values.reshape(-1, 1))

#Actual value and the predicted value
Actual_vs_predict = pd.DataFrame({'Actual value': y_test})
Actual_vs_predict['Predicted value'] = y_pred_lr
Actual_vs_predict.head()

# Accuracy
from sklearn.metrics import r2_score
LR_Model_accuracy = r2_score(y_test, y_pred_lr)


print('Making output csv for Linear model Actual vs predict values.')
df_County.to_csv(input_file_dir / "dataframe for linear model.csv")
df_County.to_csv(input_file_dir / "dataframe for Actual_vs_predict.csv")

