#!/usr/bin/env python
# coding: utf-8

# # Health Insurance Cost Prediction
# 
# Overview:
# 
# Everyone knows health insurance is crucial, but did you know several factors can affect its cost?
# In this project, I explore how various attributes influence health insurance premiums and predict costs using regression models.
# 
# Dataset Highlights:
# 
# Age: Age of the primary beneficiary.
# 
# Sex: Gender of the insurance contractor (male or female).
# 
# BMI: Body Mass Index - a key indicator of body health with an ideal range of 18.5 to 24.9.
# 
# Children: Number of kids/dependents covered by health insurance.
# 
# Smoker: Indicates whether the beneficiary smokes.
# 
# Region: Residential area in the US (northeast, southeast, southwest, northwest).
# 
# Approach:
# 
# Visualization: Graphical representation of data to highlight trends and patterns.
# 
# Regression Modeling: Multiple regression techniques are applied to predict insurance costs, enabling comparison and performance evaluation.
# 
# Dive into the code and notebooks to get a detailed perspective on health insurance cost influencers and see how different regression models fare in predicting them!
# 
# 
# 
# 
# 
# 
# 

# # Import libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # Data Preprocessing

# In[2]:


df = pd.read_csv('insurance.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[7]:


duplicated_rows = df[df.duplicated()]
print(f"Duplicated Rows:\n{duplicated_rows}")


# In[8]:


df= df.drop_duplicates()
df.shape


# # Visualization

# In[9]:


sns.distplot(df['charges'], kde=True, bins=50, color = 'g')
plt.title('Distribution of Charges')
plt.show()


# In[10]:


f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax = sns.distplot(np.log10(df['charges']), kde = True, color = 'g' )


# In[11]:


charges = df['charges'].groupby(df.region).sum().sort_values(ascending = True)
f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax = sns.barplot(charges.head(), charges.head().index, palette='Greens')


# In[12]:


plt.figure(figsize=(8, 6))
sns.barplot(x='region', y='charges', hue='sex', data=df, ci=None, palette='pastel')
plt.title('Charges by Region for Male and Female')
plt.ylabel('Average Charges')
plt.xlabel('Region')
plt.show()


# In[13]:


plt.figure(figsize=(10, 7))
sns.barplot(x='region', y='charges', hue='smoker', data=df, ci=None, palette='pastel')
plt.title('Charges by Region for Smokers and Non-Smokers')
plt.ylabel('Average Charges')
plt.xlabel('Region')
plt.show()


# In[14]:


df['children'] = df['children'].astype(str)

plt.figure(figsize=(12, 10))
sns.barplot(x='region', y='charges', hue='children', data=df, ci=None, palette='deep')
plt.title('Charges by Region Based on Number of Children')
plt.ylabel('Average Charges')
plt.xlabel('Region')
plt.legend(title='Number of Children', title_fontsize='small', fontsize='x-small')
plt.show()


# In[15]:


df[["sex", "smoker", "region"]] = df[["sex", "smoker", "region"]].apply(lambda x: pd.factorize(x)[0])
df.head()


# In[16]:


df = pd.get_dummies(df, columns=['region'], drop_first=True)


# # Checking the availablity of outliers

# In[17]:


sns.scatterplot(data=df, x='age', y='charges')
plt.title("Scatter plot of 'charge' vs 'age'")
plt.show()


# In[18]:


sns.scatterplot(data=df, x='bmi', y='charges')
plt.title("Scatter plot of 'charge' vs 'age'")
plt.show()


# In[19]:


filtered_dataset = df[(df['charges'] <= 52000) & (df['bmi'] <= 50)]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=filtered_dataset, x='age', y='charges')
plt.title("Scatter plot of 'charges' vs 'age'")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=filtered_dataset, x='bmi', y='charges')
plt.title("Scatter plot of 'charges' vs 'bmi'")
plt.show()


# # Dummy Regression 

# In[20]:


X = df.drop(['charges'], axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)


# In[22]:


dummy_predictions = dummy.predict(X_test)

mae = mean_absolute_error(y_test, dummy_predictions)
mse = mean_squared_error(y_test, dummy_predictions)
r2 = r2_score(y_test, dummy_predictions)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# # Decision Tree

# In[23]:


param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


tree = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=1)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)

best_tree = grid_search.best_estimator_
cv_mse = -grid_search.best_score_
print("Cross-validated Mean Squared Error:", cv_mse)


# # Random Forest

# In[24]:


param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


rf = RandomForestRegressor(n_estimators=100, random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=1)
grid_search_rf.fit(X_train, y_train)
print("Best hyperparameters for Random Forest:", grid_search_rf.best_params_)

best_rf = grid_search_rf.best_estimator_
cv_mse_rf = -grid_search_rf.best_score_
print("Cross-validated Mean Squared Error for Random Forest:", cv_mse_rf)


# # Gradient Boosting

# In[25]:


param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': [None, 'sqrt', 'log2']
}


gb = GradientBoostingRegressor(random_state=42)
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=1)
grid_search_gb.fit(X_train, y_train)

print("Best hyperparameters for Gradient Boosting:", grid_search_gb.best_params_)
best_gb = grid_search_gb.best_estimator_
cv_mse_gb = -grid_search_gb.best_score_
print("Cross-validated Mean Squared Error for Gradient Boosting:", cv_mse_gb)


# # Evaluating the performance of the gradient boosting algorithm

# In[26]:


gb_predictions = best_gb.predict(X_test)

test_mae = mean_absolute_error(y_test, gb_predictions)
test_mse = mean_squared_error(y_test, gb_predictions)
test_r2 = r2_score(y_test, gb_predictions)

print(f"Test MAE: {test_mae:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Test R^2: {test_r2:.2f}")


# # Comparing the actual vs. predicted values

# In[27]:


results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': gb_predictions,
    'Difference': y_test - gb_predictions
})

