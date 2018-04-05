
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
url = ('D:/My books/Data science projects/datasets/Automobile-dataset.csv')


# In[3]:


df = pd.read_csv(url , header = None)


# In[4]:


# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


# In[5]:


# Attach the column names
df.columns= headers
# List the first five rows
df.head()


# In[6]:


# checking the data type of data frame "df" by .dtypes
df.dtypes


# In[7]:


# summary of the DataFram
df.info


# Identifying and handling Missing data in our Automobile Dataset.

# In[8]:


# replacing "?" with NaN
df.replace("?", np.nan, inplace = True)
# Displaying the first 5 rows of the dataset
df.head(5)


# In[9]:


#Evaluating for missing values
missing_data = df.isnull()
missing_data.head(5)


# In[10]:


# Counting missing values in each column 
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 


# Dealing with missing values
# There are two options; 
# - dropping data
# - replacing data
# 
# Whole columns should be dropped only if most entries in the column are empty. In our dataset, none of the columns are empty enough to drop entirely.

# In[16]:


#Calculating the average of the "normalized-losses" column
avg_1 = df["normalized-losses"].astype("float").mean(axis = 0)
# Calculating the mean value for 'bore' column
avg_2=df['bore'].astype('float').mean(axis=0)
# Calculating the mean value for 'stroke' column
avg_3 = df["stroke"].astype("float").mean(axis = 0)
# Calculating the mean value for 'horsepower' column
avg_4=df['horsepower'].astype('float').mean(axis=0)
#Calculating the mean value for 'peak-rpm' column
avg_5=df['peak-rpm'].astype('float').mean(axis=0)


# In[17]:


#Replace "NaN" by mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, avg_1, inplace = True)
#Replace "NaN" by mean value in 'bore' column
df['bore'].replace(np.nan, avg_2, inplace= True)
#Replace "NaN" by mean value in 'stroke' column
df['stroke'].replace(np.nan, avg_3, inplace= True)
#Replace "NaN" by mean value in 'horsepower' column
df['horsepower'].replace(np.nan, avg_4, inplace= True)
#Replace "NaN" by mean value in 'peak-rpm' column
avg_5=df['peak-rpm'].astype('float').mean(axis=0)


# In[19]:


# To check most common type of doors
df['num-of-doors'].value_counts()


# In[20]:


#replacing the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace = True)


# In[22]:


#droping all rows that do not have price data:
df.dropna(subset=["price"], axis=0, inplace = True)
df.reset_index(drop = True, inplace = True)


# Correcting our data formats

# In[23]:


df.dtypes


# In[25]:


# Converting data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


# In[26]:


# standardizing data; transform mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]


# Calculating the correlation between variables of type "int64" or "float64"

# In[27]:


df.corr()


# Identifying linear relationship between an individual variable and the price

# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[31]:


# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)


# As the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables. Engine size seems like a pretty good predictor of price since the regression line is almost a perfect diagonal line

# In[32]:


#examining the correlation between 'engine-size' and 'price' 
df[["engine-size", "price"]].corr()


# In[33]:


#Testing with Highway mpg
sns.regplot(x="highway-mpg", y="price", data=df)


# This gives us a negative linear relationship

# In[34]:


# Peak-rpm as the predictor variable
sns.regplot(x="peak-rpm", y="price", data=df)


# Visualizing Categorical variables using boxplots

# In[37]:


# Relationship between "body-style" and "price"
sns.boxplot(x="body-style", y="price", data=df)


# In[38]:


# Relationship between "engine-location" and "price" 
sns.boxplot(x="engine-location", y="price", data=df)


#  front and rear, are distinct enough to take engine-location as a potential good predictor of price. 

# In[39]:


# drive-wheels
sns.boxplot(x="drive-wheels", y="price", data=df)


#  The distribution of price between the different drive-wheels categories differs; as such drive-wheels could potentially be a predictor of price.

# In[43]:


# Using Simple Linear Regression method to help us understand the relationship between two variables:
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

X = df[['highway-mpg']]
Y = df['price']

lm.fit(X,Y)
Yhat=lm.predict(X)
Yhat[0:5]  

# Value of the intercept
lm.intercept_


# In[44]:


# Value of the slope
lm.coef_


# Estimated linear model
# 
# Yhat=a+bX
# 
# price = 38423.31 - 821.73 x highway-mpg

# In[48]:


# Training the model using 'engine-size' as the independent variable and 'price' as the dependent variable
lm1 = LinearRegression()
lm1.fit(df[['engine-size']], df[['price']])
lm1


# In[51]:


# Slope 
lm1.coef_


# In[52]:


# Intercept
lm1.intercept_

