# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 23:57:43 2020

@author: megha
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import seaborn as sn
df=pd.read_csv(r'C:\Users\megha\Documents\BH.csv')
print(df.head)
print(df['Rating of Online Class experience'].nunique())
print(df['Medium for online class'].nunique())
df[df.isnull().values]
df['Rating of Online Class experience'] = df['Rating of Online Class experience'].replace(np.nan, 'Average')
df['Medium for online class'] = df['Medium for online class'].replace(np.nan, 'Smartphone or Laptop/Desktop')
df['Time spent on TV'].replace({'n':'0', 'N':'0', 'No tv':'0', ' ':'0', 0:'0'}, inplace = True)
df['What you miss the most'].unique()
df['What you miss the most'] = df['What you miss the most'].replace(['All the above','All of the above ','everything','All above','all of the above','ALL','all','All of the above','all of them','All of them','All '],'All')
df['What you miss the most'] = df['What you miss the most'].replace(['NOTHING','Nothing this is my usual life','To stay alone. ','Nothing ','Nah, this is my usual lifestyle anyway, just being lazy....','Normal life','My normal routine','nothing'],'Nothing')
df['What you miss the most'] = df['What you miss the most'].replace(['Only friends','Friends , relatives','relatives and friends','Family ','The idea of being around fun loving people but this time has certainly made us all to reconnect (and fill the gap if any) with our families and relatives so it is fun but certainly we do miss hanging out with friends','Family'],'Friends/Relatives/Family')

# Analyzing

%matplotlib qt
plt.figure(figsize=(12, 8))
sn.set(style='darkgrid')
sn.countplot(x='Age of Subject', data=df, palette='Set3')
plt.yscale('log')
plt.xlabel('Age of Subject', weight='bold')
plt.ylabel('Number of Subjects', weight='bold')
plt.show()

%matplotlib qt
plt.figure(figsize=(14,10))
sn.set(style='darkgrid')
sn.boxplot(data=df[['Time spent on Online Class','Time spent on self study','Time spent on fitness','Time spent on sleep','Time spent on social media', 'Time spent on TV']],orient='h', palette='Set3')
plt.yticks(weight='bold')
plt.show()


plt.figure(figsize=(12,8))
sn.set(style='darkgrid')
sn.countplot(y='Stress busters', data=df, order=df['Stress busters'].value_counts().index[:15], palette='Set3')
plt.xlabel("Number of Subjects", weight='bold')
plt.ylabel("Stress buster activity", weight='bold')
plt.show()

sn.set(style='darkgrid')
fig, ax = plt.subplots(3,2, figsize=(16,18))
sn.violinplot(x='Health issue during lockdown', y='Time spent on Online Class', data=df, ax=ax[0,0])
sn.violinplot(x='Health issue during lockdown', y='Time spent on self study', data=df, ax=ax[0,1])
sn.violinplot(x='Health issue during lockdown', y='Time spent on fitness', data=df, ax=ax[1,0])
sn.violinplot(x='Health issue during lockdown', y='Time spent on sleep', data=df, ax=ax[1,1])
sn.violinplot(x='Health issue during lockdown', y='Time spent on social media', data=df, ax=ax[2,0])
sn.violinplot(x='Health issue during lockdown', y='Time spent on TV', data=df, ax=ax[2,1])
plt.show()


corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(10,10))
sn.heatmap(corr, mask=mask, center=0, annot=True,square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


#regression 
df['Rating of Online Class experience']=df['Rating of Online Class experience'].replace('Excellent','5')
df['Rating of Online Class experience']=df['Rating of Online Class experience'].replace('Good','4')
df['Rating of Online Class experience']=df['Rating of Online Class experience'].replace('Average','3')
df['Rating of Online Class experience']=df['Rating of Online Class experience'].replace('Poor','2')
df['Rating of Online Class experience']=df['Rating of Online Class experience'].replace('Very poor','1')
df['Rating of Online Class experience']=pd.to_numeric(df['Rating of Online Class experience'])

x=df[['Time spent on Online Class']]
y=df['Rating of Online Class experience']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

  from sklearn.ensemble import RandomForestRegressor
      regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
      regressor.fit(X_train, y_train)
   y_pred = regressor.predict(X_test)
import sklearn.metrics as metrics
print(metrics.r2_score(y_test,y_pred))



#knn between x and y...acc=85.35
df['Health issue during lockdown']=df['Health issue during lockdown'].replace('YES','1')
df['Health issue during lockdown']=df['Health issue during lockdown'].replace('NO','0')
df['Health issue during lockdown']=pd.to_numeric(df['Health issue during lockdown'])


df['Change in your weight']=df['Change in your weight'].replace(['Increased','Decreased'],'1')
df['Change in your weight']=df['Change in your weight'].replace('Remain Constant','0')

X=df[['Change in your weight','Number of meals per day']]
y=df['Health issue during lockdown']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))

#============================================================================
#elbow method to find value of k
# =============================================================================
error_rate = []
# Might take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)ss
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')



knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=9')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


#logistic regression accuracy is very low
df['Rating of Online Class experience']=df['Rating of Online Class experience'].replace('Excellent','5')
df['Rating of Online Class experience']=df['Rating of Online Class experience'].replace('Good','4')
df['Rating of Online Class experience']=df['Rating of Online Class experience'].replace('Average','3')
df['Rating of Online Class experience']=df['Rating of Online Class experience'].replace('Poor','2')
df['Rating of Online Class experience']=df['Rating of Online Class experience'].replace('Very poor','1')
df['Rating of Online Class experience']=pd.to_numeric(df['Rating of Online Class experience'])


x=df[['Time spent on Online Class','Time spent on self study']]
y=df['Rating of Online Class experience']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=0)
from sklearn.linear_model import LogisticRegression
 
#Make instance/object of the model because our model is implemented as a class.
#Step 2:
LR = LogisticRegression()
 
#Train the model on the input train data
#Step 3:
LR.fit(X_train, y_train)
predictions = LR.predict(X_test)
score = LR.score(X_test, y_test)
print("Accuracy is ",score*100,"%")

numeric_col = ['Time spent on self study','Time spent on Online Class','Time spent on social media','Age of Subject','Time spent on sleep']
 
# Correlation Matrix formation
corr_matrix = df.loc[:,numeric_col].corr()
print(corr_matrix)
 
#Using heatmap to visualize the correlation matrix
sn.heatmap(corr_matrix, annot=True)
correlation_mat = df2_small.corr()

corr_pairs = corr_matrix.unstack()

print(corr_pairs)



#decision tree accuracy=69..8
df['Time utilized']=df['Time utilized'].replace('YES',1)
df['Time utilized']=df['Time utilized'].replace('NO',0)
df['Do you find yourself more connected with your family, close friends , relatives  ?']=df['Do you find yourself more connected with your family, close friends , relatives  ?'].replace('YES',1)
df['Do you find yourself more connected with your family, close friends , relatives  ?']=df['Do you find yourself more connected with your family, close friends , relatives  ?'].replace('NO',0)
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
features=['Time utilized','Time spent on social media','Age of Subject']
X=df[features]
Y=df['Do you find yourself more connected with your family, close friends , relatives  ?']
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=0)
clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)
clf_entropy.fit(X_train,y_train)
y_pred_en=clf_entropy.predict(X_test)
print(y_pred_en)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_en)*100)
pip install pydotplus
%matplotlib qt
import pydotplus
tree.plot_tree(clf_entropy)
cn=['YES','NO']

#%matplotlib qt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (3,2), dpi=500)
tree.plot_tree(clf_entropy, feature_names = features,  class_names=cn,filled = False);
fig.savefig('imagename.png')

#NEYRAL NETWORK

input = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1/(1+np.exp(-x))

from matplotlib import pyplot as plt
plt.plot(input, sigmoid(input), c="r")
import numpy as np
feature_set = df[['']]
labels = np.array([[1,0,0,1,1]])
labels = labels.reshape(5,1)





from sklearn.model_selection import train_test_split

df['Health issue during lockdown']=df['Health issue during lockdown'].replace('YES','1')
df['Health issue during lockdown']=df['Health issue during lockdown'].replace('NO','0')
print(df['Health issue during lockdown'])
df['Health issue during lockdown']=pd.to_numeric(df['Health issue during lockdown'])

features=df[['Number of meals per day','Time spent on TV','Time spent on sleep','Time spent on fitness']]
X_train,X_test, Y_train, Y_test = train_test_split(features, rating, test_size = .2, random_state = 10)
from sklearn.ensemble import RandomForestRegressor 
regressor = RandomForestRegressor()
regressor.fit(X_train, Y_train)   
Y_pred = regressor.predict(X_test) # test the output by changing values 
print(Y_pred)

import sklearn.metrics as metrics
print(metrics.r2_score(Y_test,Y_pred))
plt.scatter(X_train, Y_train, color="blue", label="original")
plt.plot(X_test, Y_pred, color="red", label="predicted")
plt.legend()
plt.show()
# random regression above r2 is =0.3 so correlation matrix

numeric_col = ['Number of meals per day','Time spent on TV','Time spent on sleep','Time spent on fitness']
corr_matrix = df.loc[:,numeric_col].corr()
print(corr_matrix)
import seaborn as sn
sn.heatmap(corr_matrix, annot=True) 


