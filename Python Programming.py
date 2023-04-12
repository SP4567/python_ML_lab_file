#!/usr/bin/env python
# coding: utf-8

# ![Alt text](c:/Users/Suyash%20Pandey/OneDrive/Desktop/python_progs/Screenshot%202022-07-29%20100309.png)
# # Name: Suyash Pandey
# # Course:B.Tech(CSE)
# # Batch: 2021-2025
# # Topic: Python Programming and Machine Learning.
# # Faculty: Dr.Namrata Dhanda.
# # Institute: Amity Institute Of Engineering And Technology.
# # College: Amity University, Lucknow, Uttar Pradesh.

# ## Introduction To Python
# #### What is Python? Executive Summary
# Python is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together. Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance. Python supports modules and packages, which encourages program modularity and code reuse. The Python interpreter and the extensive standard library are available in source or binary form without charge for all major platforms, and can be freely distributed.
# 
# Often, programmers fall in love with Python because of the increased productivity it provides. Since there is no compilation step, the edit-test-debug cycle is incredibly fast. Debugging Python programs is easy: a bug or bad input will never cause a segmentation fault. Instead, when the interpreter discovers an error, it raises an exception. When the program doesn't catch the exception, the interpreter prints a stack trace. A source level debugger allows inspection of local and global variables, evaluation of arbitrary expressions, setting breakpoints, stepping through the code a line at a time, and so on. The debugger is written in Python itself, testifying to Python's introspective power. On the other hand, often the quickest way to debug a program is to add a few print statements to the source: the fast edit-test-debug cycle makes this simple approach very effective.
# 
# 
# ### Data Types
# 
# Different data types in Python are:
# 1. int: int or the integer data type in python is used to store the numerical values.ex:x = 5
# 
# 2. float: float data type in python is used to store floating point numbers with decimal point in the numbers.ex:x = 3.5
# 
# 3. bool: bool in other languages also called boolean data type but in python is called as bool which represents only two values either True or False, in python the true and false are written with the first lettr capital. 
#          if x > y:
#          return True
#          else
#          return False
#          
# 4. String: String data type also written as str in python is used to store words,sentences in double quotes.(can be stored in single quote also). x = "hello world".
# 
# 5. char: char or the character data type in python is used to store characters in single quotation. char ch = 'a'
# 6. tuple: tuple data type is used store values separated by a comma is a collection of ordered values. google = [1, 2, 3, 4]
# 7. complex number:represented by complex class.It uis specified by (real part)+(imaginary part)j. x+yj
# 8. list: these are alomost similar to arrays but unlike arrays these can store values of different data types. 
# golu = [1, 2, "yoyo"].
#    
# 9. dictionary: unordered collection of data. dict = {India:Delhi, Australia:Canberra, USA:Washington Dc, Japan:Tokyo}.
# 
# ### Tokens
# 
# 1. Keywords: These are the special words that convey special meaning in the programming language.These are reserved for some special tasks and operations.ex: True,False,and,or etc.
# 
# 2. Identifiers: These are used to give names to different parts of python code.
# 
# 3. Operators: These are the symbols which are used to perform different operations on the operands.Operands are the values on which operation is performed. 
# 
# 4. Literals: These are those values which remains fixed during the execution of the program if they are changed then the program generates an errors.
# 
# 5. Delimitors:  A sequence of one or more characters that specifies the boundary between various sections in plain text or other 
# data streams.Python uses symbols and combinations of symbols as delimiters in expressions, literals, tuples, lists, dictionaries, strings, and various parts of a statemen

# ### Linear Regression

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
df = pd.read_csv('C:\\Users\\Suyash Pandey\\OneDrive\\Desktop\\python_progs\\boston_housing.csv')
s = df['RM']
t = df['MEDV']
plt.scatter(s,t,color = 'blue',marker='o')
plt.ylabel("MEDV")
plt.xlabel("RM")
plt.title(" Association between the predictor and the target ")
df.head(10)
X, Y = make_blobs(n_samples = 1000)
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.33)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, train_size = 0.67)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
mean_value = df['MEDV'].mean()
plt.scatter(df['RM'],df['MEDV'],color = 'blue',marker = 'o')
plt.axhline(y = mean_value,c = 'r')
plt.annotate("Mean_value",xy=(7.5,mean_value+2))
plt.xlabel("RM")
plt.ylabel("MEDV")
plt.title("Mean of RM and MEDV")
S = df[['RM']]
T = df[['MEDV']]
model = LinearRegression()
model.fit(S,T)
print("Intercept", model.intercept_)
print("Coefficient", model.coef_)
MEDV_model0 = df['MEDV'].mean()
MEDV_model1 = 10 + (12*df['RM'])
MEDV_model2 = 6 + (18*df['RM'])
df['m_model0'] = MEDV_model0
df['m_model1'] = MEDV_model1
df['m_model2'] = MEDV_model2
fig, ax = plt.subplots()
ax.scatter(x = 'RM', y = 'MEDV', data = df, color = 'blue', label = "MEDV predictor")
ax.plot(df['RM'], df['m_model0'], color = "red", label = "model0")
ax.plot(df['RM'], df['m_model1'], color = "green", label = "model1")
ax.plot(df['RM'], df['m_model2'], color = "yellow", label = "model2")
ax.set_xlabel("RM")
ax.set_ylabel("MEDV")
ax.set_title("Speculated model")
ax.legend()
plt.plot()
model0_obs = pd.DataFrame({"RM":df['RM'], "MEDV":df['MEDV'], "Actual value of MEDV":df.m_model0, "ERROR": (df.m_model0-df.MEDV)})
model0_obs
print(sum(model0_obs['ERROR']))
x = df.RM
y = df.MEDV
xiyi = x * y
l = len(df)
xmean = df.RM.mean()
ymean = df.MEDV.mean()
numerator = xiyi.sum() - l*xmean*ymean
denominator = (x**2).sum() - l*(xmean**2)
m = numerator / denominator
print("m: ", m)
c = ymean - m*xmean
print("c: ", c)
min_best_fit_model = c + m*(df.RM)
df['min_best_fit_model'] = min_best_fit_model
df[["RM", "MEDV", "min_best_fit_model"]]
fi, a = plt.subplots()
a.scatter(x = "RM", y = "MEDV",color = 'blue', data = df)
a.plot(df[['RM']], df[['min_best_fit_model']], color = "red")
a.set_ylabel("MEDV")
a.set_xlabel("RM")
a.set_title("LINE OF BEST FIT")
plt.plot()


# ### Logistic Regression

# In[3]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
st = pd.read_csv('C:\\Users\\Suyash Pandey\\OneDrive\\Desktop\\desktop files\\TATAMOTORS 2.csv')
st.head()
st.info()
st.describe()
A = st[['Profit_achieveable']]
B = st['class_profit']
plt.scatter(A,B)
plt.xlabel('Profit_achieveable')
plt.ylabel('class_profit')
X = st[['Profit_achieveable','Loss_achieveable']]
Y = st['class_profit']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.20,random_state = 0)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
model = LogisticRegression()
model.fit(X_train,Y_train)
print("Intercept:",model.intercept_)
print("Coefficient:",model.coef_)
print(model.score(X_train,Y_train))
print(model.score(X_test,Y_test))


# ### KNN

# In[4]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("C:\\Users\\Suyash Pandey\\OneDrive\\Desktop\\defaulter.csv")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_to_scale = ["balance", "income"]
scaled_values = scaler.fit_transform(df[features_to_scale])
df["norm_balance"] = scaled_values[:,0]
df["norm_income"] = scaled_values[:,0]
df.head
from sklearn.model_selection import train_test_split
X = df[['norm_balance','norm_income']]
Y = df['defaulter']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=100)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3,metric="euclidean")
model.fit(X_train,Y_train)
train_accuracy = model.score(X_train,Y_train)
test_accuracy = model.score(X_test,Y_test)
print(train_accuracy)
print(test_accuracy)
train_accuracies = []
test_accuracies = []
k_vals = [i for i in range(1,100)]
features = ["norm_balance","norm_income"]
target = "defaulter"
for k in k_vals:
    model = KNeighborsClassifier(n_neighbors=k,metric='euclidean')
    model.fit(X_train,Y_train)
    train_accuracy_k = model.score(X_train,Y_train)
    test_accuracy_k = model.score(X_test,Y_test)
    train_accuracies.append(train_accuracy_k)
    test_accuracies.append(test_accuracy_k)
plt.plot(k_vals,train_accuracies)
plt.plot(k_vals,test_accuracies)
plt.legend(['train_accuracies','test_accuracies'])


# ### Decision Trees

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
cd = pd.read_csv("C:\\Users\\Suyash Pandey\\OneDrive\\Desktop\\desktop files\\credit_risk.csv") 
cd.info()
X = cd.columns.drop("class")
y = cd['class']
cd_enco = pd.get_dummies(cd[X])
print("total number of predictors after encoding = ", len(cd_enco.columns))
cd_enco.columns
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(cd_enco,y,test_size=0.15,random_state=100)
print("Shape of X_train and y_train are:", X_train.shape, "and", y_train.shape, " respectively")
print("Shape of X_train and y_train are:", X_test.shape, "and", y_test.shape, " respectively")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state = 1)
model.fit(X_train,y_train)
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test) 
from sklearn.tree import export_graphviz
import graphviz
data = export_graphviz(model, out_file=None, feature_names=cd_enco.columns,class_names=model.classes_,) 
graph = graphviz.Source(data) 
print(graph)
train_accuracy = model.score(X_train,y_train)
print("Accuracy of the model on train data = ",train_accuracy)
test_accuracy = model.score(X_test,y_test)
print("Accuracy of the model on test data = ",test_accuracy)
model1 = DecisionTreeClassifier(min_samples_split=10,min_impurity_decrease=0.005)
model1.fit(X_train,y_train)
print("train_accuracy = ", model1.score(X_train,y_train))
print("test_accuracy = ", model1.score(X_test,y_test))
model2 = DecisionTreeClassifier(min_samples_split=20,min_impurity_decrease=0.1)
model2.fit(X_train,y_train)
print("Model2 train accuracy = ", model2.score(X_train,y_train))
print("Model2 test accuracy = ", model2.score(X_test,y_test))


# ### Clustering

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
md = pd.read_csv("C:\\Users\\Suyash Pandey\\OneDrive\\Desktop\\mnist_data.csv")
im = np.asarray(md.iloc[0:1,:]).reshape(28,28)
plt.imshow(im,cmap=plt.cm.gray)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=10)
model.fit(md)
print(np.unique(model.labels_))
cluster1 = md[model.labels_==0]
cluster1_imgs = cluster1.iloc[[np.random.randint(0,cluster1.shape[0])
                              for i in range(0,5)]]
for i in range(0,cluster1_imgs.shape[0]):
    plt.subplot(1,5,i+1)
    img_fig = np.asarray(cluster1_imgs[i:i+1]).reshape(28,28)
    plt.imshow(img_fig,cmap=plt.cm.gray)
cluster2 = md[model.labels_==1]
cluster2_imgs = cluster2.iloc[[np.random.randint(0,cluster2.shape[0])
                              for i in range(0,5)]]
for i in range(0,cluster2_imgs.shape[0]):
    plt.subplot(1,5,i+1)
    img_fig = np.asarray(cluster2_imgs[i:i+1]).reshape(28,28)
    plt.imshow(img_fig,cmap = plt.cm.gray)


# ### Naive Bayes

# In[2]:


from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target 
# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
# training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train) 
# making predictions on the testing set
y_pred = gnb.predict(X_test) 
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)


# In[3]:


pip install pandoc


# In[ ]:




