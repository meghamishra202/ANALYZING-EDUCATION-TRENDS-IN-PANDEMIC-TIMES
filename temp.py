# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import seaborn as sb
df=pd.read_csv(r'C:\Users\megha\Documents\googleplaystore.csv')


#to see how many apps are available in different categories
plt.figure(figsize=(65,10))
%matplotlib qt
x=sb.countplot(x='Category', data=df)
x.set_xticklabels(x.get_xticklabels(),rotation=90)

df1=df.dropna()
df[['Rating']].to_numpy()
#df2=df.iloc[0:49,0:12]


gh=df[df.Category=='EDUCATION']
gh=gh.dropna()
g= gh['Installs']
gh['last']=g.str[-1:]
gh['Installs']= gh.Installs.str.replace('[+,#]','')
print(gh['Installs'])
for value in gh['last']:
    if (value=='k'):
        gh['Installs']=gh.Installs.replace('[k]','')
        gh['Installs'] = gh['Installs'].apply(lambda x: x*1000)
del df['last']
gh['Installs']=gh['Installs'].astype(int)
gh.drop_duplicates()

print(df2)
large=gh.nlargest(10,"Rating")
large
#highest rating apps in art category
large.plot(kind='scatter',x='App',y='Installs')
large=df2.nlargest(1,"Rating")
large
df3=df.iloc[98:139,0:12]
print(df3)
large3=df3.nlargest(1,"Rating")
large3
df4=df.iloc[139:187,0:12]
print(df4)
large4=df3.nlargest(1,"Rating")
large4
data=[large,large3,large4]
r= pd.concat(data)
print(r)
df6=df.iloc[188:297,0:12]
print(df6)
df7=df.iloc[298:335,0:12]
large5=df7.nlargest(1,"Rating")
df8=df.iloc[477:699,0:12]
large6=df8.nlargest(1,"Rating")
df8=df.iloc[477:699,0:12]
large7=df8.nlargest(1,"Rating")
df10=df.iloc[856:1004,0:12]
large8=df10.nlargest(1,"Rating")
data=[large,large3,large4,large5,large6,large7,large8]
r= pd.concat(data)
print(r)
#highest rating app of various categories
%matplotlib qt
r.plot(kind='bar',x='Category',y='Rating')

l=df[df.Category=='Education']
aa=l.nlargest(5,"Rating")

k=df[df.Category=='FINANCE']
bb=k.nlargest(1,"Rating")
j=df[df.Category=='FOOD_AND_DRINK']
cc=j.nlargest(1,"Rating")
i=df[df.Category=='HEALTH_AND_FITNESS']
dd=i.nlargest(1,"Rating")
h=df[df.Category=='HOUSE_AND_HOME']
ee=h.nlargest(1,"Rating")
g=df[df.Category=='LIBRARIES']
ff=g.nlargest(1,"Rating")
f=df[df.Category=='LIFESTYLE']
gg=f.nlargest(1,"Rating")
e=df[df.Category=='GAME']
hh=e.nlargest(1,"Rating")
d=df[df.Category=='FAMILY']
ii=d.nlargest(1,"Rating")
c=df[df.Category=='MEDICAL']
jj=c.nlargest(1,"Rating")
b=df[df.Category=='SOCIAL']
kk=b.nlargest(1,"Rating")
a=df[df.Category=='SHOPPING']
ll=a.nlargest(1,"Rating")
data=[large,large3,large4,large5,large6,large7,large8,aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk,ll]
r= pd.concat(data)
print(r)

#highest rating app of various categories
%matplotlib qt

aa.plot(kind='bar',x='Category',y='Rating')
import matplotlib.pyplot as plot

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
a=df[df.Category=='EDUCATION']

dataFrame = pd.DataFrame(data=a, columns=['Installs','Size'])


dataFrame.plot.scatter(x='Installs', y='Size', title= "Scatter plot between two variables X and Y")

#Scatterplot for relationship between price and no of installs
plt.figure(figsize=(200,10))
dataFrame = pd.DataFrame(data=a, columns=['Price','Installs'])
dataFrame.plot.scatter(x='Price', y='Installs', title= "Scatter plot");

df2=df.iloc[0:49,0:12]

df3=df.iloc[98:139,0:12]
df4=df.iloc[139:187,0:12]
df6=df.iloc[188:297,0:12]
df7=df.iloc[298:335,0:12]
df8=df.iloc[477:699,0:12]
df8=df.iloc[477:699,0:12]
df10=df.iloc[856:1004,0:12]
k=df[df.Category=='FINANCE']
j=df[df.Category=='FOOD_AND_DRINK']
i=df[df.Category=='HEALTH_AND_FITNESS']
h=df[df.Category=='HOUSE_AND_HOME']
g=df[df.Category=='LIBRARIES']
f=df[df.Category=='LIFESTYLE']
e=df[df.Category=='GAME']
d=df[df.Category=='FAMILY']
c=df[df.Category=='MEDICAL']
b=df[df.Category=='SOCIAL']
a=df[df.Category=='SHOPPING']
print(df.dtypes)
index_names = df[ df['Reviews'] == '3.0M' ].index 
df.drop(index_names, inplace=True)
df['Reviews']=pd.to_numeric(df.Reviews)

gh=df[df.Category=='EDUCATION']





data2= [df2,df3,df4,df6,df7,df8,df10,k,j,i,h,g,f,e,d,c,b,a]

# scatterplot between raqtings and price of various categories
import seaborn as sb
sb.scatterplot(x='', y='Rating', hue='Category', data=df)

#boxplot
g=df[df.Category=='EDUCATION']
d3 = g[(g['Rating']>=3) & (g['Rating']<5)]
sb.boxplot('Rating', 'Installs', data=d3)
kk=d3.nlargest(5,"Rating")
#facet graph  
t = sb.FacetGrid(kk, col='App')
t = t.map(sb.kdeplot, 'Rating')

#nahi hua
 r_df=df[df['Rating']>4.7]
 r_df['App']=r_df.index
 co=r_df.columns
 print(co)
 print(r_df)
import seaborn as sb
#sb.barplot(x='App', y='Rating', hue='Category')

# educational apps 
df2= pd.read_csv(r'C:\Users\megha\Documents\app_reviews.csv')
print(df2)
df2.head()
import nltk
from nltk.corpus import stopwords
print(df2.dtypes)
# Create stopword list:

from wordcloud import WordCloud, STOPWORDS

e=df2[df2.name=='WhiteHatJr']
name=df2.groupby("name")
#top 5 highest thumbsup count of every app
t=name.mean().sort_values(by="thumbsUpCount",ascending=False).head()


df2= pd.read_csv(r'C:\Users\megha\Documents\app_reviews.csv')

# wordclud for white ht jr
c=df2[df2.name=='WhiteHatJr']
stopwords= set(STOPWORDS)
stopwords.add("laptop")
stopwords.add(" give")
stopwords.add("class")
text=c.content
t = " ".join(review for review in text)
print(len(t))
wordcloud = WordCloud(background_color="white", stopwords=stopwords).generate(t)
%matplotlib qt

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

uniqueValues = df2['name'].unique()
print(uniqueValues)

#wordcloud for khan academy

d=df2[df2.name=='Toppr']
stopwords= set(STOPWORDS)
stopwords.add("want")
stopwords.add(" will")
stopwords.add("use")
stopwords.add("really")
stopwords.add("please")
tex=d.content
g = " ".join(review for review in tex)
print(len(g))
wordcloud = WordCloud(background_color="white", stopwords=stopwords).generate(g)
%matplotlib qt

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


name = df2.groupby("name")
plt.figure(figsize=(15,10))
name.max().sort_values(by="thumbsUpCount",ascending=False)["thumbsUpCount"].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Highest point of Wines")
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
#documrnt term matrix
a=df2[df2.name=='Toppr']
d1=a.content
g = " ".join(review for review in d1)
print(len(g))


# logistic model
a.dropna(inplace=True)
a = a[a['score']!= 3]
a['Positively_Rated'] = np.where(a['score']>3, 1, 0)
a.head(10)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(a['content'], a['Positively_Rated'], random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)
print(X_train_vectorized)
from sklearn.linear_model import LogisticRegression,SGDClassifier
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
from sklearn.metrics import roc_curve, roc_auc_score, auc
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)


## SVM 
from sklearn.model_selection import train_test_split
a.dropna(inplace=True)
a = a[a['score']!= 3]
a['Positively_Rated'] = np.where(a['score']>3, 1, 0)
a.head(10)

X_train, X_test, y_train, y_test = train_test_split(a['content'], a['Positively_Rated'], random_state=0)
from sklearn.svm import SVC

svclassifier=SVC(kernel='linear')
svclassifier.fit(X_train_vectorized,y_train)
y_pred=svclassifier.predict(vect.transform(X_test))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, y_pred))





#converting + in installs to threee 000
gh=df
gh=gh.dropna()
g= gh['Installs']
gh['last']=g.str[-1:]
gh['Installs']= gh.Installs.str.replace('[+,#]','')
print(gh['Installs'])
for value in gh['last']:
    if (value=='k'):
        gh['Installs']=gh.Installs.replace('[k]','')
        gh['Installs'] = gh['Installs'].apply(lambda x: x*1000)
del df['last']
gh['Installs']=gh['Installs'].astype(int)
df['Installs'] = np.where(df['Installs']>50000, 1, 0)


from sklearn import preprocessing
  
from sklearn.preprocessing import LabelEncoder
#
# Instantiate LabelEncoder
#
le = LabelEncoder()
#
# Encode single column status
#
df.Category = le.fit_transform(df.Category)
# column= type. if free then 0 if paid then 1
print(df['Category'])
x=df.iloc[:,1:2].values
print(x)
y=df.iloc[:,5]
print(y)



from sklearn.ensemble import RandomForestRegressor 
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
  
 # create regressor object 
regressor = RandomForestRegressor() 
  
# fit the regressor with x and y data 
regressor.fit(X_train, y_train)   
Y_pred = regressor.predict(X_test) # test the output by changing values 
print(Y_pred)

plt.scatter(X_train, y_train, color="blue", label="original")
plt.plot(X_test, Y_pred, color="red", label="predicted")
plt.legend()
plt.show()
# Visualising the Random Forest Regression results 

# arange for creating a range of values 
# from min value of x to max 
# value of x with a difference of 0.01 
# between two consecutive values 
X_grid = np.arange(min(x), max(x), 0.01) 

# reshape for reshaping the data into a len(X_grid)*1 array, 
# i.e. to make a column out of the X_grid value				 
X_grid = X_grid.reshape((len(X_grid), 1)) 

# Scatter plot for original data 
plt.scatter(x, y, color = 'blue') 

# plot predicted data 
plt.plot(X_grid, regressor.predict(X_grid), 
		color = 'green') 
plt.title('Random Forest Regression') 
plt.xlabel('Category') 
plt.ylabel('Installs') 
plt.show()
import sklearn.metrics as metrics
mae = metrics.mean_absolute_error(Y_test, Y_pred)
mse = metrics.mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse) # or mse**(0.5)  
r2 = metrics.r2_score(y_test,Y_pred)

print("Results of sklearn.metrics:")
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)
 
# multiple regression rating ~ reviews+installs+type

print(df.dtypes)
df['Reviews']=df['Reviews'].astype('int')
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
rating=df['Rating']
features=df[['Installs','Type','Reviews']]
X_train,X_test, Y_train, Y_test = train_test_split(features, rating, test_size = .2, random_state = 10)
regr = LinearRegression()
regr.fit(X_train, Y_train)
print('\nIntercept: ',regr.intercept_)
pd.DataFrame(data = regr.coef_, index = X_train.columns, columns=['coef'])
print('the r-squared Training Data: ', regr.score(X_train, Y_train))
print('the r-squared Testing Data: ', regr.score(X_test, Y_test))

Y_pred = regr.predict(X_test)
print(metrics.mean_squared_error(Y_test,Y_pred))
print(metrics.r2_score(Y_test,Y_pred))

#random forest regression on the same
rating=df['Rating']
features=df[['Installs','Type','Reviews']]
X_train,X_test, Y_train, Y_test = train_test_split(features, rating, test_size = .2, random_state = 10)
from sklearn.ensemble import RandomForestRegressor 
regressor.fit(X_train, Y_train)   
Y_pred = regressor.predict(X_test) # test the output by changing values 
print(Y_pred)
print(metrics.r2_score(Y_test,Y_pred))
plt.scatter(X_train, Y_train, color="blue", label="original")
plt.plot(X_test, Y_pred, color="red", label="predicted")
plt.legend()
plt.show()

#logistic

from sklearn.feature_extraction.text import CountVectorizer
# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)
print(X_train_vectorized)
from sklearn.linear_model import LogisticRegression,SGDClassifier
model = LogisticRegression()
model.fit(X_train_vectorized, Y_train)
from sklearn.metrics import roc_curve, roc_auc_score, auc
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
df2['Type']=df2['Type'].replace('Free',0)
df2['Type']=df2['Type'].replace('Paid',1)
print(df['Type'])


df['Size']=df['Size'].str.replace('[k]','')
df['Size']=df['Size'].replace('Varies with device', '1')
df['Size']= df.Size.str.replace('[+,#]','')
h=df['Size']
df['l']=h.str[-1:]
for value in df['l']:   
    if(value=='M'):
        df['Size']=df['Size'].str.replace('[M]','')
        df['Size'] = df['Size'].apply(lambda x: x*1000)

df['Size']= df.Size.str.replace('[+,#]','')




#category wise installs

from sklearn import preprocessing
  
from sklearn.preprocessing import LabelEncoder
#
# Instantiate LabelEncoder
#
le = LabelEncoder()
#
# Encode single column status
#
gh.Category = le.fit_transform(gh.Category)
print(df['Installs'])
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
import statsmodels.api as sm 
x=df.iloc[:,1:2].values
print(x)
y=df.iloc[:,5]
print(y)
from sklearn.ensemble import RandomForestRegressor 
x=df.Type.values.reshape(-1,1)
y=df.Installs.values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)
RandomForestRegModel=RandomForestRegressor()
RandomForestRegModel.fit(x_train,y_train)
y_pred=RandomForestRegModel.predict(x_test)
print(y_pred)
r2 = metrics.r2_score(y_test,y_pred)
print(r2)

#kmeans

x=gh[['Category','Installs','Rating']]
from sklearn.cluster import KMeans

WCSS=[]
for i in range(1,14):
    model_km=KMeans(n_clusters=i)
    model_km.fit(x)
    WCSS.append(model_km.inertia_)
plt.plot(list(range(1,14)),WCSS,marker='o')
plt.xlabel("optimum")
plt.ylabel("wcss")

model_km=KMeans(n_clusters=4)
kmeans=model_km.fit(x)


plt.scatter(x['Installs'],x['Category'],c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
plt.show()

model_km=KMeans(n_clusters=3)
model_km.fit(x)
cn=model_km.predict(x)
print(cn)

# correlation matrix

import os
import pandas as pd
import numpy as np
import seaborn as sn
 
# Loading the dataset
# Numeric columns of the dataset
numeric_col = ['reviewCreatedVersion','thumbsUpCount','score']
 
# Correlation Matrix formation
corr_matrix = df2.loc[:,numeric_col].corr()
print(corr_matrix)
 
#Using heatmap to visualize the correlation matrix
sn.heatmap(corr_matrix, annot=True)
correlation_mat = df2_small.corr()

corr_pairs = corr_matrix.unstack()

print(corr_pairs)
sorted_pairs = corr_pairs.sort_values(kind="quicksort")

print(sorted_pairs)

strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]

print(strong_pairs)


#df2 correlation
cf=pd.DataFrame(df2,columns=['Type','Installs','Rating','Reviews'])
print(cf)
corr_matrix=cf.corr()
print(corr_matrix)
print(df2['Reviews'].dtype)
df2['Reviews']=df2['Reviews'].astype(int)


print(df2.columns)


numeric_col = ['Type','Installs','Rating','Reviews','Category']
 
# Correlation Matrix formation
corr_matrix = df2.loc[:,numeric_col].corr()
print(corr_matrix)
 
#Using heatmap to visualize the correlation matrix
sn.heatmap(corr_matrix, annot=True)
correlation_mat = df2.corr()

corr_pairs = corr_matrix.unstack()

print(corr_pairs)
sorted_pairs = corr_pairs.sort_values(kind="quicksort")

print(sorted_pairs)

strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]

print(strong_pairs)





df2=df[df["Category"]=="EDUCATION"]
print(df2)








#analysis
pip install plotly
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot
import plotly.offline as py