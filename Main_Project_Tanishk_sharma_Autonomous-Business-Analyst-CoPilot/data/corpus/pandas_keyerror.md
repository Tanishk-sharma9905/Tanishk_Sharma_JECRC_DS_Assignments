# Pandas KeyError Operations

## KeyError: 'column_name'
**Cause**: The code referenced a column name that does not exist in the DataFrame. This is commonly caused by:
- Case mismatches (e.g. `'sales'` vs `'Sales'`).
- Leading or trailing whitespaces in the CSV/Excel headers (e.g. `'revenue '` vs `'revenue'`).
- Spaces vs underscores (e.g. `'Order ID'` vs `'order_id'`).
- Hallucinated column names that are not in the dataset schema.

**Correction**: Inspect the actual column names in the dataset and use them exactly as they are defined.
```python
# To fix: list all columns to verify spelling and casing
print(df.columns.tolist())

# Strip whitespaces from column names if headers are messy:
df.columns = df.columns.str.strip()

# Access column using exact spelling:
df['Correct_Column_Name']
```
