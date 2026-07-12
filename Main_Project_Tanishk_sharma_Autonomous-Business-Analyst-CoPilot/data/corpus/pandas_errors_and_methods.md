# Pandas Common Errors and Operations

## AttributeError: 'DataFrame' object has no attribute 'append'
**Cause**: The `.append()` method was deprecated in pandas 1.4.0 and completely removed in pandas 2.0.0.
**Correction**: Use `pd.concat` to combine DataFrames.
Instead of:
```python
df = df.append(new_row, ignore_index=True)
```
Use:
```python
new_df = pd.DataFrame([new_row])
df = pd.concat([df, new_df], ignore_index=True)
```

## ValueError: cannot reindex on an axis with duplicate labels
**Cause**: Attempting to align or merge data frames that have duplicate labels on the index.
**Correction**: Drop duplicates or reset the index.
```python
df = df.reset_index(drop=True)
# or
df = df[~df.index.duplicated(keep='first')]
```

## SettingWithCopyWarning
**Cause**: Modifying a view slice of a DataFrame instead of a copy.
**Correction**: Use `.loc` for assignment, or explicitly call `.copy()` when slicing.
```python
# Instead of: df_subset['col'] = values
df_subset = df[['col1', 'col2']].copy()
df_subset['col1'] = values
```

## Datetime Parsing
Convert a column to datetime safely:
```python
df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
```

## Grouping and Aggregation
```python
# Multi-column aggregation
df.groupby('Category').agg({
    'Revenue': 'sum',
    'Quantity': 'mean'
}).reset_index()
```
