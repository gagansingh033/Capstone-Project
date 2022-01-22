# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:49:28 2021

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

# chceking shape of dataframe
df.shape

# Descibing the dataset with stat
df.describe()

# Rename the columns
df.rename(columns={'TotalPop': 'TotalPopulation'}, inplace=True)

# 1.	Which county of state counted for higher population?

# Grouping county for calculations
df_County = df.groupby('County').sum()

df_county = pd.DataFrame()
for i in df["County"].unique():
    counties_df = df[df['County'] == i]
    counties_df.loc[df.index[-1] + 1] = counties_df.sum()
    counties_df['State'] = list(counties_df["State"][:1])[0]
    counties_df['County'] = list(counties_df["County"][:1])[0]
    df_county = df_county.append(counties_df[-1:])

df_county = df_county.copy(deep=True).reset_index(drop=True)

higher_population = df_county[['State','County','TotalPopulation','Men','Women']].loc[df_county['TotalPopulation'] == df_county['TotalPopulation'].max()]


class High_population():
    def __init__(self):
        self.state = higher_population.State.values[0]
        self.county = higher_population.County.values[0]
        self.totalpopulation = higher_population.TotalPopulation.values[0]
        self.men = higher_population.Men.values[0]
        self.women = higher_population.Women.values[0]
       
        
Hp = High_population()    
print('Q1 :- Which county of state counted for higher population?')
print('Ans 1 :- The {} County in {} State has maximum population with {} count,\
      This count conatin {} men and {} women. \n'.format(Hp.county, Hp.state, Hp.totalpopulation, Hp.men, Hp.women))


# Use all models in each question 
# 2. Prediction of population and the income of men and women by county?
print('\n Q2 :- Prediction of population and the income of men and women by county?')
print(' Running Linear model for prediction from LR.py module')
exec(open('LR.py').read())

# 3.	In which counties most people eligible for voting?
top_5_states = df_county['VotingAgeCitizen'].nlargest(5)
eligible_voters = pd.DataFrame()
for i in top_5_states:
    eligible_voting_top5 = df_county[['State','County','VotingAgeCitizen']].loc[df_county['VotingAgeCitizen'] == i]
    eligible_voters = eligible_voters.append(eligible_voting_top5)

print('\n Q3 :- In which counties most people eligible for voting?')
print('Ans3 :- Here are the top 5 Counties who conatins high voters')
print( eligible_voters.reset_index(drop=True))

# 4. Determine the highest ratio of child poverty in the US?
print('\n Q4:-Determine the highest ratio of poverty with income in the US?')
print('Running Decision tree and Random forest model for this question')
exec(open('DT.py').read())

df_accuracy = pd.DataFrame({'Model Names': pd.Categorical(["Linear regression model accuracy", "Decision tree model accuracy","Random forest model accuracy"]), 
                            'Accuracy': (Model_accuracy, DT_Model_accuracy, RF_Model_accuracy)})

df_accuracy['Accuracy'] = round(df_accuracy['Accuracy']*100,2)
