#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import seaborn as sns


# In[6]:


df=pd.read_csv("D:/data analytics/E-Commerce Sales, Product & Customer Analysis/E-Commerce Sales Dataset.csv")


# In[7]:


df.head()


# In[8]:


p=df.columns
print(p)


# In[9]:


# Data Cleaning
df.isnull().sum()
df['rating'].fillna(df['rating'].mean(),inplace=False)
df['discount_percent'].fillna(0,inplace=False)
df['listing_date'] = pd.to_datetime(df['listing_date'], dayfirst=True)
#df['listing_date']=pd.to_datetime(df['listing_date'])
df.rename(columns={"seller_city":"City"},inplace=True)
df.rename(columns={"discount_percent":"discount %"},inplace=True)


# In[10]:


#feature Engineering

df['revenue']=df['final_price'] * df['units_sold']
df['delivery_bucket']=pd.cut(
    df['delivery_days'],
    bins=[0,3,7,15,30],
    labels=['Fast','Medium','Slow','Very Slow']
)
print(df[['delivery_days', 'delivery_bucket']].head(10))


# In[11]:


#EDA-Explorator Data Analysis

#Revenue BY Category

revenue_category=df.groupby('category')['revenue'].sum().div(10000000).round(2).sort_values(ascending=False)
df.groupby('category')['revenue'].sum()
#plot
plt.figure(figsize(12,7))
bx=revenue_category.plot(kind='bar',color='Lightgreen')
plt.title("Revenue By Category")
plt.xlabel('Category')
plt.ylabel('revenue')
plt.xticks(rotation=45)
# Add value labels
for container in bx.containers:
    bx.bar_label(container, fmt='%.0f', fontsize=8)
plt.tight_layout()
plt.show()


# In[12]:


#Discount By Sales

df[['discount %','units_sold']].corr()
plt.scatter(df['discount %'],df['units_sold'])
plt.xlabel('Discount %')
plt.ylabel('Units Sold')
plt.title('Discount Vs Units Sold')
plt.show()


#  Electronics contribute highest revenue
#  Discounts positively impact sales up to a limit
#  Faster delivery improves ratings
#  Few top sellers generate majority revenue
#  Pune has the highest Revenue


# In[14]:


#CUSTOMER ANALYSIS QUESTIONS

# 1.customer Demotography
city_sales = (
    df.groupby('City')['units_sold']
      .sum()
      .div(100000)        # Convert to Lakhs
      .round(2)
      .sort_values(ascending=False)
)

ax = city_sales.plot(kind='bar', figsize=(10,5))

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f L', padding=3)

plt.ylabel("Units Sold (Lakhs)")
plt.title("City Vs UnitSold")
plt.show()



# In[15]:


# 2.Revenue By City

df['revenue'] = df['final_price'] * df['units_sold']
city_revenue=df.groupby('City')['revenue'].sum().round(2).div(1000000).sort_values(ascending=False)
df.groupby('City')['revenue'].sum()
# Plot
plt.figure(figsize=(12,6))
ax = city_revenue.plot(kind='bar', color='skyblue')
plt.title('Revenue by City')
plt.xlabel('City')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', fontsize=8)
plt.tight_layout()
plt.show()


# In[16]:


#Customer Purchasing Behavior

   #Preferred Payment Mode

c=df.groupby('payment_modes')['units_sold'].sum().round(2).div(1000)
print('perferred payment mode:',c)


# In[17]:


# Delivery Experience Impact on Customers

  #Delivery Days vs Rating

df.groupby('delivery_days')['rating'].mean()
plt.scatter(df['delivery_days'], df['rating'])
plt.xlabel("Delivery Days")
plt.ylabel("Rating")
plt.title("Delivery Time vs Customer Rating")
plt.show()

#Key Insight: Longer delivery → lower satisfaction.


# In[18]:


#RETURNS AND CUSTOMER PERFERENCE

 #Returnable vs Non-Returnable

returnable=df.groupby('is_returnable') ['units_sold'].sum()
print(returnable)

# Key Insight: Customer prefer returnable products


# In[19]:


corr = df[['final_price','discount %','units_sold','rating','delivery_days']].corr()

#Correlation Heatmap

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Customer Behavior Correlation Matrix")
plt.show()

# Key Insights:
# Final Price and Discount have a moderate negative relationship (-0.33).
# All other variables have very weak or no linear correlation (values close to 0).
# Product price does not significantly affect sales volume.
# Discounts are not strongly associated with higher sales.
# Delivery time has little impact on ratings.
# Ratings are largely independent of price and discounts.


# In[31]:


#City-wise Customer Demand

city_count = (
    df['City']
      .value_counts()
      .head(10)
      .reset_index()
)

city_count.columns = ['City', 'Customer_Count']

plt.figure(figsize=(15,5))

ax = sns.barplot(
    data=city_count,
    x='Customer_Count',
    y='City'
)

for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3)

plt.title("Top 10 Cities by Customer Demand")
plt.xlabel("Customer Count")
plt.ylabel("City")

plt.show()


# In[21]:


# City Vs Delivery Days

s=df.groupby('City')['delivery_days'].mean().sort_values().round(2)
print(s)
# Final Price VS Rating

sns.histplot(
    x='rating',
    y='final_price',
    data=df,
)
plt.show()


# In[22]:


# How has revenue changed over time?

monthly = (
    df.groupby(df['listing_date'].dt.to_period('M'))['revenue']
      .sum()
      .reset_index()
)
monthly['listing_date'] = monthly['listing_date'].astype(str)
monthly['revenue_billion'] = (monthly['revenue'] / 1e9).round(2)

plt.figure(figsize=(14,6))
plt.plot(
    monthly['listing_date'],
    monthly['revenue_billion'],
    marker='o',
    linewidth=2
)

plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Revenue (₹ Billion)")
plt.title("Monthly Revenue Trend")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# In[23]:


#Which brands are driving revenue?
brand_rev =df.groupby('brand')['revenue'].sum().div(100000).round(2).sort_values(ascending=False)
print("brand_revenue",brand_rev)


# In[24]:


#Which cities have the fastest delivery?
df.groupby('City')['delivery_days'].mean()


# In[25]:


# which seller are performing the best?
seller_rev = (
    df.groupby('seller')['revenue']
      .sum()
      .div(100000)      # Converting to Lakhs
      .round(2)
      .sort_values(ascending=False)
)

print(seller_rev)


# In[26]:


#Which products receive the most reviews?
df.groupby('product_name')['review_count'].sum().head(10)


# In[ ]:





# In[ ]:




