# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:09:38 2021

@author: Gagan Saini
"""


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics

# Importing Requried libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image  
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


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
df['Poverty_in_num'] = [percent("{}*{}%".format(x,y)) for x, y in zip(df['TotalPopulation'], df['Poverty'])]



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

df = df_county[['TotalPopulation','Income','Poverty_in_num']]

# after dividing into the train and test do these steps
# if else if 
df['Income_levels'] = 0
df.loc[df['Income'] >= 30000, 'Income_levels'] = 0
df.loc[df['Income'] >= 80000, 'Income_levels'] = 1
df.loc[df['Income'] >= 200000, 'Income_levels'] = 2

df['Poverty_levels'] = 0
df.loc[df['Poverty_in_num'] >= 30000, 'Poverty_levels'] = 0
df.loc[df['Poverty_in_num'] >= 80000, 'Poverty_levels'] = 1
df.loc[df['Poverty_in_num'] >= 200000, 'Poverty_levels'] = 2

df['Total_population_levels'] = 0
df.loc[df['TotalPopulation'] >= 5000, 'Total_population_levels'] = 0
df.loc[df['TotalPopulation'] >= 25000, 'Total_population_levels'] = 1
df.loc[df['TotalPopulation'] >= 50000, 'Total_population_levels'] = 2
df.loc[df['TotalPopulation'] >= 500000, 'Total_population_levels'] = 3
df.loc[df['TotalPopulation'] >= 1000000, 'Total_population_levels'] = 4
# chceking shape of dataframe
df.shape
df.head()
# Descibing the dataset with stat
df.describe()
df.to_csv(input_file_dir / "dataframe for Decision Tree.csv")
#split dataset in features and target variable
feature_cols = ['Income_levels','Poverty_levels']
X = df[feature_cols] # Features
y = df.Total_population_levels #df.Income_levels # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

DT_Model_accuracy = metrics.accuracy_score(y_test, y_pred)
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols, class_names=['0','1','2','3','4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Model_accuracy = float(0.8798)
graph.write_png('Poverty_levles_with_class.png')
Image(graph.create_png())


#########################################################
# Random Forest
forest=RandomForestClassifier()
forest.fit(X_train,y_train)
y_pred=forest.predict(X_test)

print("accuracy of Random forest classifier model:", forest.score(X_train,y_train))
RF_Model_accuracy = forest.score(X_train,y_train)
################################ confusion_matrix

confusion_matrix_result = confusion_matrix(y_test, y_pred)
ax=sns.heatmap(confusion_matrix_result/np.sum(confusion_matrix_result),annot=True,fmt='.2%',cmap='Blues')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values');


def make_confusion_matrix_graph(var):
    plt.figure(var)
    df_confusion_matrix=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
    df_confusion_matrix = df_confusion_matrix.loc[df_confusion_matrix['Actual'] ==  var]
    confusion_matrix_result = confusion_matrix(df_confusion_matrix['Actual'], np.array(df_confusion_matrix['Predicted']))
    ax=sns.heatmap(confusion_matrix_result/np.sum(confusion_matrix_result),annot=True,fmt='.2%',cmap='Blues')
    ax.set_title('\nConfusion Matrix for Actual Values - {}'.format(str(var)))
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values');


graph_0 = make_confusion_matrix_graph(0)
graph_1 = make_confusion_matrix_graph(1)
graph_2 = make_confusion_matrix_graph(2)
graph_3 = make_confusion_matrix_graph(3)
graph_4 = make_confusion_matrix_graph(4)
