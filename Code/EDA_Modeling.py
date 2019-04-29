# -*- coding: utf-8 -*-
"""
Data Capstone Project:  Low-Interaction Honeypot Traffic: Modeling the Search
Data:                   Rapid7 Heisenberg Cloud Honeypot cowrie Logs

Exploratory Data Analysis (EDA), Statistical Tests, & Modeling
"""

# Import packages
import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare, chi2_contingency, poisson
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import export_graphviz
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import confusion_matrix, roc_auc_score
import time

 
'''
Data Import:


Import data from SQL Server Management Studio (SSMS).
Confirm correct data types for all variables.
Check top and bottom 5 rows.

 '''


# Set variables
server = 'LAPTOP-3PQN51FL'
database = 'Heisenberg_Honeypot'

# Connect to SSMS
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=' + server + ';'
                      'Database=' + database + ';'
                      'Trusted_Connection=yes;')

query = 'SELECT * FROM Heisenberg_Honeypot.dbo.FINAL'
df = pd.read_sql(query, conn)


# Check data types
for i in df:
    print(i, df[i].dtype)
    
# Fix ts
df['ts'] = pd.to_datetime(df['ts'])

print(df.head())
print(df.tail())
print(df.columns)

# Delete unneeded variables - Clear up kernel
del server
del database
del query
  

'''
Basic EDA Visualizations:

Create bar charts for each variable.
Create scatterplots for ts variable.
Create bar chart and scatterplots with dependent variable (dst_port) and frequency counts of the other variables.
Create categorical plots (catplots) with dependent variable (dst_port) and src_ip, eventid, sensor, Country, & City.

'''


# Take a small random sample of the data available to visualize distributions.
n = 1000
df_sample = df.sample(n = n,random_state = 0)


# Count Plots (Bar charts)
for i in df_sample:
    sns.set(style="darkgrid")
    ax = sns.countplot(x = df_sample[i], data = df_sample)
    plt.xticks(rotation = 45)
    #plt.savefig(i+' count.png')
    plt.tight_layout()
    plt.show()

# Check variables over time
    # Timestamp ('ts') scatterplot
target = df_sample['ts']
left = min(target)
right = max(target)

for i in df_sample:
    plt.scatter(target, df_sample[i])
    plt.xlim(left, right)
    plt.xlabel('Timestamp')
    plt.ylabel(i)
    plt.xticks(rotation = 90)
    #plt.savefig('Scatter_ts_' + i + '.png')
    plt.show()


# Group by Target Variable for further analysis: 'dst_port'
target = df_sample['dst_port']
target_unique = df_sample.groupby(target).nunique()
target_unique.drop('dst_port', axis = 1, inplace = True)


# Bar Charts
for i in target_unique:
    plt.bar(target_unique.index, target_unique[i])
    plt.xlabel('dst_port')
    plt.ylabel(i)
    #plt.savefig('Bar_dst_port_' + i + '.png')
    plt.show()

# Scatterplots
for i in df_sample:
    plt.scatter(target,df_sample[i])
    plt.xlabel('dst_port')
    plt.ylabel(i)
    plt.xticks(rotation = 90)
    #plt.savefig('Scatter_dst_port_' + i + '.png')
    plt.show()
    
    
# Catplots - Three variables (X, y, hue)
    # Select specific relationships based on results
'''
ax = sns.catplot(x = df_sample['dst_port'], y = df_sample['Country'], hue = 'eventid', data = df_sample)
plt.xticks(rotation = 90)
#plt.savefig('dst_port_Country_Catplot.png')
plt.tight_layout()
plt.show()    

ax = sns.catplot(x = df_sample['dst_port'], y = df_sample['eventid'], hue = 'Country', data = df_sample)
plt.xticks(rotation = 90)
#plt.savefig('dst_port_eventid_Catplot.png')
plt.tight_layout()
plt.show()   

'''
# Delete unneeded variables - Clear up kernel
del left
del right
del target_unique


'''
Statistical Testing:

Chi Square Test of Independence
Chi Square Goodness of Fit Test
Poisson Cumulative Distribution Function
Covariance Matrix
'''


# Chi Square Test of Independence 
    # Returns chi2 test statistic, p-value, dof, expected ndarray
    # Null hypothesis is the variables/categories are dependent (no difference).

# Set the sample size and pull a random sample from the data set
    # Sample size selected with online calculator.  Size is sufficient for 95% CI with 1% margin of error.
n = 9497 
df_sample = df.sample(n = n, random_state = 0)


# Break down by variable, do a nunique on the categories within each variable
p_list = []
for i in df:
    df_unique = df.groupby(i).nunique()
    for j in df_unique:
        obs = np.array(df_unique)
        chi2, p, dof, expected = chi2_contingency(obs)
        if p < 0.05:
            line = [i, j, p, chi2]
        p_list.append(line)
        
print(p_list)

# Drop variables that are not independent
df.drop(['ts', 'session'], axis = 1, inplace = True)

# Rerun df_sample 
df_sample = df.sample(n = n, random_state = 0)

# Poisson Distributions
    # Cumulative Distribution Function (CDF) shows the probability weights associated with each variable
    # when compared with the target variable (dst_port).

# CDF requires object, not datetime.
#df_sample['ts'] = str(df_sample['ts'])

# Set the target and get a frequency chart of the other variables.
target = df_sample['dst_port']
df_unique = df_sample.groupby(target).nunique()
df_unique.drop('dst_port', axis = 1, inplace = True)

# Save the column names from the data set (subcategories).
col = []
for i in df_unique.index:
    indexName = i
    col.append(indexName)

# Set bin number to match number of subcategories.
bins_num = len(col)

# Transpose the dataframe and capture the number of categories in dst_port.
df_unique = df_unique.T
var_bin = len(df_unique.index)

# Cumulative Distribution Function (Probability Heatmap)
rate = 1
poisson_cdf = poisson.cdf(df_unique, rate)
cdf_df = pd.DataFrame(poisson_cdf, columns = df_unique.columns, index = df_unique.index)
cdf_heat = sns.heatmap(cdf_df)
plt.title('Probability Distribution')
plt.ylabel('Variables')
#plt.savefig('poisson_cdf.png')

# Poisson Probability Histogram - Higher values have a higher probability %
plt.hist(poisson_cdf,bins = var_bin)
plt.title('CDF Histogram')
plt.xlabel('Destination Ports')
#plt.savefig(i+' CDF_hist.png')
plt.show()


# Chi-Square Goodness of Fit Test 
    # Returns chi square test statistic & p-value
    # Null hypothesis is that the categorical data doesn't differ statistically from a normal distribution.
        # All must reject the null hypothesis, p-values are > 0.05; none are normal.


# One hot encode the df_sample
df_sample = pd.get_dummies(df_sample)   

# Loop through each variable and conduct the test.
results = []
for i in df_sample:
    line = {i:chisquare(df_sample[i])}
    results.append(line)

# Check any results that return a p-value less than 0.05.
for i in results:
    for j,k in i.items():
        if k[-1] < 0.05:
            print(j,k)  
            

# Covariance Analysis
    # This will calculate the covariance of the standardized data elements.
        # The closer the covariance value is to zero, the more independent the variables are.

# Reset df_sample for calculation
n = 9497 
df_sample = df.sample(n = n, random_state = 0)

# Initialize and prep variables
target = 'dst_port'
features = ['src_ip', 'eventid', 'sensor', 'Country', 'City']

X = df_sample[features]
y = df_sample[target].values

# Encode the variables: LabelEncoder for y, One Hot Encoding for X's
le = LabelEncoder()
y = le.fit_transform(y)
X = pd.get_dummies(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Set the scaler and normalize the X variables
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

# Determine columns to features #4,954
# print(X.columns[:3726]) # src_ip
# print(X.columns[3726:3729]) # eventid
# print(X.columns[3729:3787]) # sensor
# print(X.columns[3787:3894]) # Country
# print(X.columns[3894:4954]) # City

# Calculate the covariance matrix
np_cov = np.cov(X_train.T)


# Plot the Covariance Matrix 
ax = sns.heatmap(np_cov)
plt.title('Covariance Matrix')
#plt.savefig('covariance_matrix.png')


# Delete unneeded variables - Clear up kernel
del bins_num
del cdf_df
del chi2
del col
del df_unique
del dof
del expected
del i
del indexName
del j
del k
del line
del np_cov
del obs
del p
del p_list
del poisson_cdf
del rate
del results
del var_bin


''' 
Modeling 

    - Dependent Var: 'dst_port' 
    - Independent Vars: 'src_ip', 'eventid', 'sensor', 'Country', 'City'
                        
Models Included:
    - Random Forest Classifier
    - Hyperparameter Tuning to determine optimal model between:
        - Logistic Regression
        - Support Vector Machine (SVM)
        - Multinomial Naive Bayes
    - Best Model Selection & Run
'''


# Variable Initialization

# Set the variables for all models and create numpy arrays for optimal processing.

# Reset df_sample for testing
n = 9497
df_sample=df.sample(n=n,random_state=0)

# Load the variables.  
    # Note features and target are already loaded from previous section.
X = df_sample[features]
y = df_sample[target].values


# Encode the variables: LabelEncoder for y, One Hot Encoding for X's
y = le.fit_transform(y)
X = pd.get_dummies(X)

#Oversampling to balance the data for machine learning models.
ros = RandomOverSampler(random_state = 0)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.3, random_state = 0)

# Normalize the X variables
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

# -----------------------------------------------------------------------------
# Random Forest Classifier
    # This is an ensemble learning method that creates a group of decision trees and merges them together.
    
# Load the model
rfc = RandomForestClassifier()

# Fit the model with training sets
rfc.fit(X_train, y_train)

# Save top estimators to create tree graphic.
estimator = rfc.estimators_[0]

# Score the model with 10 fold cross validation and get the accuracy.
scores = cross_val_score(rfc, X_test, y_test, cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0))
print(scores)

# Average the splits for overall score.
score = round(scores.mean(), 2)
print(score)

# Save the feature_importance to find variables with highest information gain.
importances = rfc.feature_importances_
f_importances = pd.Series(importances, X.columns)
f_importances = f_importances.sort_values(ascending = False)
print(f_importances)


# Grab original dst_port values (unique) and save the names.
y = df_sample[target].values
y_names = np.unique(y)

# Create RFC Tree graph and translate to .png file.
    #* Once ran, copy the textfile information.  
    # Go to https://dreampuf.github.io/GraphvizOnline/ and paste to generate the image.
    # Right click, save image as 'rfc.png' to save the graphic.
export_graphviz(estimator, out_file = 'rfc.txt', 
                feature_names = X.columns,
                class_names = y_names, max_depth = 5,
                rounded = True, rotate = True, proportion = False, 
                precision = 2, filled = True)

# -----------------------------------------------------------------------------
# Hyperparameter Tuning is used to find the ideal combination of model and parameters for
    # best accuracy scores.  Chose Logistic Regression, SVM, and Multinomial Naive Bayes.


# Split a small sample for parameter selection. 
    # n = 1,000 takes 1.5 hours to run
    # n = 100 takes 11.4 seconds to run
    # Both sample sizes return the same model and parameters.
df_samp = df.sample(n = 100, random_state = 0)


# Get the target and feature vectors
X = df_samp[features]
y = df_samp[target].values

# Encode the variables: LabelEncoder for y, One Hot Encoding for X's
y = le.fit_transform(y)
X = pd.get_dummies(X)

#Oversampling to balance.
X_resampled, y_resampled = ros.fit_resample(X, y)

# Create a dictionary of the chosen models.  
clfs = {'lr': LogisticRegression(random_state = 0),
        'svm': SVC(random_state = 0),
        'nb' : MultinomialNB()}
        

n_components = [X.shape[1] // 4, X.shape[1] // 2, X.shape[1]]


# Create a Pipeline
pipe_clfs = {}

for name, clf in clfs.items():
    pipe_clfs[name] = {}
    if name == 'nb':
        for n_component in n_components:
            pipe_clfs[name][n_component] = Pipeline([('clf', clf)])
    else:
        for n_component in n_components:
            pipe_clfs[name][n_component] = Pipeline([('StandardScaler', StandardScaler()), 
                                                 ('clf', clf)])
    
# Create a dictionary for parameter specifications based on each model.
param_grids = {}


# Logistic Regression parameters
C_range = [10 ** i for i in range(-4, 5)]

param_grid = [{'clf__multi_class': ['multinomial'],
               'clf__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
               'clf__C': C_range}]

param_grids['lr'] = param_grid



# SVM parameters
param_grid = [{'clf__C': [0.01, 0.1, 1, 10, 100],
               'clf__gamma': [0.01, 0.1, 1, 10, 100],
               'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]

param_grids['svm'] = param_grid

# Multinomial Naive Bayes parameters
param_grid = [{'clf__alpha': [0.01, 0.1, .5, 1],
               'clf__fit_prior': [True, False]}]

param_grids['nb'] = param_grid


# Run for each classifier with specified variety of parameters to determine the best combination.
best_score_param_estimators = []

start = time.time()

for name in pipe_clfs.keys():
    for n_component in n_components:   
        gs = GridSearchCV(estimator=pipe_clfs[name][n_component],
                          param_grid=param_grids[name],
                          scoring='accuracy',
                          n_jobs=-1,
                          cv=StratifiedKFold(n_splits = 10,
                                             shuffle = True,
                                             random_state = 0))
        gs = gs.fit(X_resampled, y_resampled)

        best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
end = time.time()

# Sort best_score_param_estimators in descending order of the best score.
best_score_param_estimators = sorted(best_score_param_estimators, key = lambda x : x[0], reverse = True)

# Print the timer.
print(end - start) # 11.4 seconds

# Print out best_estimator, parameters, and accuracy.
print(best_score_param_estimators[0][2])
print("Accuracy: ", best_score_param_estimators[0]) 


# -----------------------------------------------------------------------------

# Winning Model Run: Multinomial Naive Bayes
    # Hyperparameter model assessed 92% accuracy on sample size 100.
    # Scores for sample size 15,000: Cross validate x10 - average accuracy is 94.11%. 1 hr to train & test full data set.
    # Sample size 1,000: Cross validate x10 - average accuracy is %


# One large batch run to generate interpretable visualizations (confusion matrix & classification report)
n = 15000
df_sample = df.sample(n = n,random_state = 0)

# Reset the variables.
X = df_sample[features]
y = df_sample[target].values

# Encode the variables: LabelEncoder for y, One Hot Encoding for X's
y = le.fit_transform(y)
X = pd.get_dummies(X)

#Oversampling to balance.
X_resampled, y_resampled = ros.fit_resample(X, y)

# Split data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.3, random_state = 0)


# Declare the classifier dynamically with hyperparameter tuning results.
pipe_clf = best_score_param_estimators[0][2]


# Fit the pipeline.
start = time.time()
mnb_fit = pipe_clf.fit(X_train, y_train)
end = time.time()
print(end - start) #  seconds


# Score the model with 10 fold cross validation and get the accuracy.
scores = cross_val_score(pipe_clf, X_test, y_test,  cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0))
#print(scores)

# Average the splits for overall score.
score = round(scores.mean(), 4)
print(score)

# Get y_pred for further analysis
y_pred = mnb_fit.predict(X_test)


# Confusion Matrix
# A table used to describe the performance of the classification model's 
# predictions with the true target values.

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False, cmap=plt.cm.Blues)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('Confusion Matrix')
plt.tight_layout()
# plt.savefig('confusionmatrix.png')

# Classification Report
    # Displays the precision, recall, and F1 score for each class in the y variable.
    
# Create and plot the classification report.
viz = ClassificationReport(pipe_clf, classes=y_names)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.poof()

# -----------------------------------------------------------------------------

# Initiate batch loop to avoid memory errors for full model run.

# Save scores and times to list for overall averaging and visualization.
model_results = [] 
auc_list = []
time_list = []

# Determine number of loops required to cover all of the df when sample size = 1,000.
    # Add 1 to capture last iteration - not a full 15,000.
length = int(len(df)/1000 + 1)
    
    
# Run batch loop to obtain model scores.

start = time.time()
for i in range(length):

    # Set the variables for all models and create numpy arrays for optimal processing.
    if i < (length - 1):
        n = 1000
        df_sample = df.sample(n = n,random_state = 0)
        df = df.drop(df_sample.index)
    else:
        df_sample = df
    
    # Reset the variables.
    X = df_sample[features]
    y = df_sample[target].values
    
    # Encode the variables: LabelEncoder for y, One Hot Encoding for X's
    y = le.fit_transform(y)
    X = pd.get_dummies(X)
    
    #Oversampling to balance.
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Split data into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.3, random_state = 0)
    
    
    # Declare the classifier dynamically with hyperparameter tuning results.
    pipe_clf = best_score_param_estimators[0][2]
    
    
    # Fit the pipeline.
    start = time.time()
    mnb_fit = pipe_clf.fit(X_train, y_train)
    end = time.time()
    timer = end - start
    time_list.append(timer)
    
    
    # Score the model with 10 fold cross validation and get the accuracy.
    scores = cross_val_score(pipe_clf, X_test, y_test,  cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0))
    #print(scores)
    
    # Average the splits for overall score.
    score = round(scores.mean(), 4)
    print(score)
    model_results.append(score)
    
    # Get y_pred for further analysis
    y_pred = mnb_fit.predict(X_test)
    
    # AUC Calculation    
    #Create one-versus-all binary classes for y_test
    lb = LabelBinarizer()
    lb.fit(y_test)
    lb.fit(y_pred)
    y_test_bin = lb.transform(y_test)
    y_pred_bin = lb.transform(y_pred)
    
    # Compute AUC Score - 0.9908592283598858
    auc = roc_auc_score(y_test_bin, y_pred_bin, average="macro")
    auc_list.append(auc)
    
end = time.time()
print(end-start)

# Average Accuracy Scores
results_array = np.array(model_results)
final_result = round(results_array.mean(),4)
print(final_result * 100)


# Average AUC Scores
auc_array = np.array(auc_list)
final_auc = round(auc_array.mean(),4)
print(final_auc * 100)

# Check run times
print(time_list)
