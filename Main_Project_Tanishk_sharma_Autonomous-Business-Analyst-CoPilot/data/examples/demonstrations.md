# Autonomous Data Science Co-Pilot Demonstrations

This document provides 5 complete demonstration examples as requested in the project brief. To replicate these, simply create a mock dataset in the required format and upload it into the Streamlit app.

## 1. Sales Dashboard
**Input Dataset:** A `sales.csv` with columns: `Date`, `Region`, `Product`, `Revenue`.
**User Prompt:** "Build a sales dashboard showing revenue by region and a trend over time."
**Generated Code Output:** 
```python
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

df = pd.read_csv('workspace/default_project/uploads/sales.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Revenue by Region
region_sales = df.groupby('Region')['Revenue'].sum().reset_index()
fig1 = px.bar(region_sales, x='Region', y='Revenue', title='Revenue by Region')
fig1.write_html('revenue_by_region.html')

# Trend Over Time
time_trend = df.groupby('Date')['Revenue'].sum().reset_index()
fig2 = px.line(time_trend, x='Date', y='Revenue', title='Sales Trend Over Time')
fig2.write_html('sales_trend.html')
```
**Output Displayed:** Two interactive Plotly charts rendered directly in Streamlit, along with key insights explaining which region performed best.

## 2. Data Quality Audit
**Input Dataset:** An `employees.xlsx` with many missing values and duplicate records.
**User Prompt:** "Run a data quality audit on this file. Impute missing numeric values with the mean, drop duplicate rows, and give me the cleaned dataset."
**Generated Code Output:** 
```python
import pandas as pd
df = pd.read_excel('workspace/default_project/uploads/employees.xlsx')

initial_shape = df.shape
df.drop_duplicates(inplace=True)

# Impute numeric
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

df.to_csv('cleaned_employees.csv', index=False)
```
**Output Displayed:** A downloadable `cleaned_employees.csv` button appears in the Streamlit UI, and the text explains how many rows were dropped.

## 3. Trend Analysis
**Input Dataset:** A `traffic.tsv` recording daily website hits.
**User Prompt:** "Is my traffic growing? Plot a 7-day rolling average."
**Generated Code Output:**
```python
import pandas as pd
import plotly.express as px

df = pd.read_csv('workspace/default_project/uploads/traffic.tsv', sep='\t')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Rolling_Avg'] = df['Hits'].rolling(window=7).mean()

fig = px.line(df, x='Date', y=['Hits', 'Rolling_Avg'], title='Traffic Trend with 7-Day Average')
fig.write_html('traffic_trend.html')
```
**Output Displayed:** An interactive Plotly chart with dual lines, highlighting the macro trend.

## 4. Customer Cohort Analysis
**Input Dataset:** `customers.parquet` with columns `Customer_ID`, `Sign_Up_Date`, `Total_Spend`.
**User Prompt:** "Auto-segment customers by total spend into High, Medium, and Low tiers using quartiles, and visualize the segments."
**Generated Code Output:**
```python
import pandas as pd
import plotly.express as px

df = pd.read_parquet('workspace/default_project/uploads/customers.parquet')
df['Spend_Tier'] = pd.qcut(df['Total_Spend'], q=3, labels=['Low', 'Medium', 'High'])

tier_counts = df['Spend_Tier'].value_counts().reset_index()
tier_counts.columns = ['Tier', 'Count']
fig = px.pie(tier_counts, values='Count', names='Tier', title='Customer Segments by Spend')
fig.write_html('cohort_segments.html')
```
**Output Displayed:** An interactive pie chart outlining the size of each cohort.

## 5. Ad-hoc Queries
**Input Dataset:** A `logistics.sqlite` database file containing a table of shipments.
**User Prompt:** "Which carrier has the highest average delay? Show me a bar chart."
**Generated Code Output:**
```python
import sqlite3
import pandas as pd
import plotly.express as px

conn = sqlite3.connect('workspace/default_project/uploads/logistics.sqlite')
df = pd.read_sql("SELECT Carrier, Delay_Minutes FROM shipments", conn)

delay_by_carrier = df.groupby('Carrier')['Delay_Minutes'].mean().reset_index().sort_values('Delay_Minutes', ascending=False)
fig = px.bar(delay_by_carrier, x='Carrier', y='Delay_Minutes', title='Average Delay by Carrier')
fig.write_html('carrier_delay.html')
```
**Output Displayed:** A sorted interactive bar chart answering the operational query instantly without requiring SQL or Python knowledge from the user.
