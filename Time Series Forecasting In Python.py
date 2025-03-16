#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 


# In[3]:


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.linear_model import LinearRegression


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


df =pd.read_csv("gold_monthly_csv.csv")


# In[6]:


df


# In[7]:


print(df)


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


print(f"data range of gold price available from - {df.loc[0,'Date']} to {df.loc[len(df)-1,'Date']}")


# In[11]:


Date = pd.date_range (start ='1/1/1950' , end='8/1/2020')


# In[12]:


Date


# In[13]:


df = df.rename(columns={'Date': 'month'})  


# In[14]:


df


# In[23]:


df.plot(figsize=(20, 8))
plt.title("Gold prices monthly since 1950 and onwards")
plt.xlabel("Months")
plt.ylabel("Price")
plt.show()


# In[16]:


round(df.describe(),3)


# In[18]:


df.describe()


# In[26]:


# Boxplot
plt.figure(figsize=(20, 8))  # Create a new figure
sns.boxplot(x='month', y='Price', data=df)
plt.title("Gold prices monthly since 1950 and onwards (Boxplot)")
plt.xlabel("Months")
plt.ylabel("Price")
plt.grid(True)
plt.show()

#If you want to remove the month column after the plot.
df.drop('month', axis=1, inplace=True)


# In[29]:


from statsmodels.graphics.tsaplots import month_plot


# In[48]:


fig, ax = plt.subplots(figsize=(18, 8))
month_plot(df['Price'], ylabel='Gold Price', ax=ax)
plt.title("Gold prices monthly since 1950 and onwards")
plt.xlabel('Month')
plt.ylabel('Price')
plt.grid(True)
plt.show()


# In[61]:


df = df.sort_index()

# Sample Data (replace with your actual data)
dates = pd.to_datetime(pd.date_range('2020-01-01', periods=24, freq='M'))
data = {
    'Price': np.random.rand(24) * 100,
    'Volume': np.random.randint(100, 1000, 24),
    'Sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], 24)
}
df = pd.DataFrame(data, index=dates)

# Multiple Graph Design
fig, axes = plt.subplots(3, 2, figsize=(15, 12))  # 3 rows, 2 columns

# 1. Time Series Plot (Price)
df['Price'].plot(ax=axes[0, 0], title='Price Over Time', color='skyblue')
axes[0, 0].set_ylabel('Price')
axes[0, 0].grid(True, linestyle='--', alpha=0.7)

# 2. Time Series Plot (Volume)
df['Volume'].plot(ax=axes[0, 1], title='Volume Over Time', color='salmon')
axes[0, 1].set_ylabel('Volume')
axes[0, 1].grid(True, linestyle='--', alpha=0.7)

# 3. Histogram (Price)
sns.histplot(df['Price'], ax=axes[1, 0], kde=True, color='lightgreen')
axes[1, 0].set_title('Price Distribution')

# 4. Boxplot (Volume)
sns.boxplot(x=df['Volume'], ax=axes[1, 1], color='lightcoral')
axes[1, 1].set_title('Volume Boxplot')

# 5. Bar Plot (Sentiment Counts)
sentiment_counts = df['Sentiment'].value_counts()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=axes[2, 0], palette='viridis')
axes[2, 0].set_title('Sentiment Distribution')
axes[2, 0].set_ylabel('Count')

# 6. Scatter Plot (Price vs. Volume)
axes[2, 1].scatter(df['Volume'], df['Price'], color='purple', alpha=0.7)
axes[2, 1].set_title('Price vs. Volume')
axes[2, 1].set_xlabel('Volume')
axes[2, 1].set_ylabel('Price')
axes[2, 1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[72]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
file_path = "gold_monthly_csv.csv"
df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

# Ensure the dataset is sorted by date
df = df.sort_index()

# Convert data to numeric and handle missing values
df = df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, setting errors to NaN
df = df.dropna()  # Remove any missing values

# Ensure correct column selection for price (assumes first column is price data)
price_column = df.columns[0]  # Automatically selects the first column

# Prepare data for machine learning
df['Month'] = df.index.month
df['Year'] = df.index.year
X = df[['Month', 'Year']]
y = df[price_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Forecast gold prices for 2025
forecast_steps = 12  # Predict next 12 months dynamically
forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
forecast_data = pd.DataFrame({'Month': forecast_index.month, 'Year': forecast_index.year})
forecast_prices = model.predict(forecast_data)

# Convert forecast to a DataFrame
forecast_df = pd.DataFrame({'Predicted Price': forecast_prices}, index=forecast_index)

# Plot forecasted prices
plt.figure(figsize=(10, 5))
plt.plot(df[price_column], label='Historical Price')
plt.plot(forecast_df['Predicted Price'], label='Forecasted Price', linestyle='dashed', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Gold Price Forecast for 2025')
plt.legend()
plt.show()

# Print forecasted values
print("Forecasted Gold Prices for 2025:")
print(forecast_df)


# In[ ]:




