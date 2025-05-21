#Access database 

import sqlite3
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,roc_auc_score,precision_score
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
db = sqlite3.connect("data/bmarket.db")
cursor = db.cursor()

cursor.execute('SELECT * FROM bank_marketing;')
rows = cursor.fetchall()
numpy_array = np.array(rows)

value_replacements = {'Cell':'cellular','Telephone':'telephone',"unknown": None,999:0,"yes":1,"no":0}
for row in numpy_array:
    if row[9]< 0:
        row[9] = -row[9]
    row[1] = int(row[1][0:2])
    for old,new in value_replacements.items():
        row[row == old] = new
dataframe = pd.DataFrame.from_records(numpy_array,columns = ("ID","Age","Occupation","Marital Status","Education Level","Credit Default","Housing Loan","Personal Loan","Contact Method","Campaign Calls","Previous Contact Days","Subscription Status")) # np.array to pd.DataFrame


dataframe.drop(['ID'], axis=1, inplace=True)





feature_cols = ["Age","Occupation","Contact Method","Campaign Calls","Previous Contact Days"]
x =dataframe[feature_cols] # Features
y =dataframe["Subscription Status"] # Target variable
x = pd.get_dummies(x) 
y= pd.get_dummies(y) 




# PCA 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
print(X_scaled[:4])

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)
print(X_pca[:4])
print("Explained variance:", pca.explained_variance_ratio_)
print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))



# Split dataset into training set and test set

x_train, x_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object

clf = tree.DecisionTreeClassifier(max_depth=12)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)




# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
# Model Accuracy, how often is the classifier correct?


# Fit the classifier to the training data
clf.fit(x_train, y_train)

# Make predictions
y_pred = clf.predict(x_test)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred,target_names=["no","yes"])
gini = roc_auc_score(y_test, y_pred)

# Print the results

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
print("Gini Report:", gini)

