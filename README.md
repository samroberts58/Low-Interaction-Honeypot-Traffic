# Low-Interaction Honeypot Traffic: Modeling an Attacker’s Search
Samantha Roberts, M.S. Data Science <br/>
George Washington University

## Introduction

### Purpose & Goals:

This is a data science capstone project for graduate-level evaluation.  The purpose is to solve a real-world problem by applying data science techniques and project management skills to create a functional solution. 


Honeypots are a cybersecurity defensive tool used to deflect intrusive attempts and record attacker behaviors. This project utilizes network traffic recorded from a network of low-interaction honeypots as the foundation for a real-world cybersecurity problem.  Many private and public entites may experience intrusive cybersecurity attacks but lack resources for an elaborate high-interaction honeypot solution.  For these organizations, using a low-interaction honeypot may be a more feasible approach.  The goal of this project is to find actionable intelligence from low-interaction honeypots as a budget-friendly response to cybersecurity needs.  It is hoped that successful trend modeling and predictive analysis can assist with identifying an entity's targeted entry locations and common tactics to provide the basis for security system hardening best practices.


### Data Set Information:

The data set is "Rapid7 Heisenberg Cloud Honeypot cowrie Logs" and was retrieved from https://opendata.rapid7.com/heisenberg.cowrie/.

The original zip file is 19.9 GB and contains 13 JSON files with over 10 million total records.  The data captured is an extract from their honeypot network, deployed globally, and covers a datetime range from November 1st, 2016 through November 30th, 2016. Main variables include source port, destination port, honeypot machine names (pseudonyms), and type of connection/interaction requested.

## Methodology

### Tools & Software:

Python:
	Python 3.6 <br/>
		- Intel Distribution for Python (IDP) used as Python Interpreter 	--not required <br/>
	Packages Required:
```
import json
import codecs
import csv
import os
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
```
CSV <br/>
Microsoft SQL Server Management Studio (SSMS) Version 18.0 <br/>
Tableau Desktop Version 2019.1.2 <br/>
	*Tableau Online account required for viewing visualizations.


### Files Included:

Python:
```
	Heisenberg_toCSV.py
	EDA_Modeling.py
```
JSON:
```
	1_0_0.json
	1_1_0.json
	1_2_0.json
	1_3_0.json
	1_4_0.json
	1_5_0.json
	1_10_0.json
	1_11_0.json
	1_12_0.json
	1_13_0.json
	1_14_0.json
	1_15_0.json
	1_16_0.json
```
SQL:
```
	CreateFinaltbl.sql
	DataPreprocessing.sql
```
CSV:
```
	src_ip_lookup.csv
	CSV_1_0_0.csv
	CSV_1_1_0.csv
	CSV_1_2_0.csv
	CSV_1_3_0.csv
	CSV_1_4_0.csv
	CSV_1_5_0.csv
	CSV_1_10_0.csv
	CSV_1_11_0.csv
	CSV_1_12_0.csv
	CSV_1_13_0.csv
	CSV_1_14_0.csv
	CSV_1_15_0.csv
	CSV_1_16_0.csv
```
Textfile (instructions, commentary):
```
	Data_Preprocessing.txt
	EDA_Modeling.txt
	ChiSquareInd_results.txt
	rfc.txt
	WebsiteContents.txt
```
HTML
```
	index.html
	overview.html
	analysis.html
	conclusion.html
	about.html
```
CSS
```
	style.css
	bootstrap.min.css
```
Java Script (JS):
```
	jquery-3.3.1.min.js
	script.js
	images.js
```
Tableau Workbook:
```
	DATS6501-Capstone.twb
```


### Process Flow

1) Download the data from Rapid7's website, and unzip to the intended working directory for Python.
2) Run 'Heisenberg_toCSV.py'.
3) Process the CSV files in acccordance with steps designated in Data_Preprocessing.txt.
4) Create 'Heisenberg_honeypot' database and import each CSV file into SQL Server Management Studio (SSMS).
5) Join the data using CreateFinaltbl.sql.
6) Process the data using DataPreprocessing.sql.
7) Run 'EDA_Modeling.py' <br/>
	*Recommended to run in sections: <br/>	
			&ensp;&ensp;&ensp;&ensp;Import data from SSMS. <br/>
			&ensp;&ensp;&ensp;&ensp;Basic EDA Visualizations <br/>
			&ensp;&ensp;&ensp;&ensp;Statistical Testing <br/>
			&ensp;&ensp;&ensp;&ensp;Modeling: Variable Initialization through Random Forest Classifier <br/>
			&ensp;&ensp;&ensp;&ensp;Modeling: Hyperparameter Tuning <br/>
			&ensp;&ensp;&ensp;&ensp;Modeling: Best Model Run <br/>
	*Additional information included in EDA_Modeling.txt.


## Model Design & Techniques:

Statistical testing included: <br/>
		- Chi-Square Goodness of Fit <br/>
		- Chi-Square Test of Independence <br/>
		- Probability Distribution Function <br/>
		- Covariance Matrix <br/>

Models: <br/>
		- Random Forest Classifier <br/>
		- Logistic Regression <br/>
		- Support Vector Machine (SVM) <br/>
		- Multinomial Naive Bayes Classifer <br/>

Model Visualizations: <br/>
		- Classification Report (includes precision, recall, F1 score, support) <br/>
		- Confusion Matrix <br/>
		
Hyperparameter Tuning was used to assist in determining the optimal combination of model and parameters among the chosen classification models (Logistic Regression, SVM, and Multinomial Naive Bayes).


## Acknowledgements

Code from George Washington University's Data Science Machine Learning 1 Course Exercise 12 was used extensively in assistance with the hyperparameter tuning portion of the script.

## License 
[MIT](https://choosealicense.com/licenses/mit/)