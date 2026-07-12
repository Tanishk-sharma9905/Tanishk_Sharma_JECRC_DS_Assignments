# Plotly Express Common Errors and Operations

## ValueError: Invalid element(s) received for the 'x' property of scatter
**Cause**: The column name passed to `x`, `y`, or `color` parameters does not exist in the DataFrame, is misspelled, has casing mismatches, or has incorrect types.
**Correction**: Verify column names using `df.columns.tolist()` before plotting.
```python
# Check column names first:
print(df.columns)
# Correct usage:
fig = px.scatter(df, x='Correct_Column_Name', y='Other_Column')
```

## ValueError: could not convert string to float
**Cause**: Trying to plot a categorical or string column on a numeric/continuous axis or using it for a continuous aggregation inside a plotly metric.
**Correction**: Aggregate the data first or cast the column to numeric:
```python
# Instead of: px.line(df, x='Date', y='Categorical_String')
df_agg = df.groupby('Date')['Numeric_Value'].sum().reset_index()
fig = px.line(df_agg, x='Date', y='Numeric_Value')
```

## Plotly Saving/Exporting in Sandbox
**Cause**: Streamlit checks for `.html` or `.png`/`.jpg` files in the current workspace directory to render charts to the user. Returning a Plotly figure object without writing it to disk is not shown.
**Correction**: Save Plotly charts as HTML in the current directory.
```python
import plotly.express as px
fig = px.bar(df, x='Category', y='Sales', title='Sales by Category')
fig.write_html('chart.html')
```
