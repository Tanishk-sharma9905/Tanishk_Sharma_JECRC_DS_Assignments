# Scikit-learn Common Errors and Operations

## ValueError: Input X contains NaN
**Cause**: KMeans, LinearRegression, or StandardScaler was called on data that contains missing (NaN/null) values.
**Correction**: Drop missing values or impute them before scaling or training:
```python
# Drop rows with NaN in features
df_clean = df[cols].dropna()
# Or impute missing values
df_clean = df[cols].fillna(df[cols].mean())

# Scale and fit
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)
```

## ConvergenceWarning: Number of distinct clusters found smaller than n_clusters
**Cause**: The requested number of clusters (`n_clusters`) is larger than the number of unique/distinct records in the training dataset.
**Correction**: Reduce `n_clusters` to a value smaller than the unique record count:
```python
unique_points = len(df[cols].drop_duplicates())
n_clusters = min(3, unique_points)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
```

## Standard KMeans and Scaling Template
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Select numeric features & drop missing values
cols = ['Feature_A', 'Feature_B']
df_clust = df[cols].dropna()

# 2. Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clust)

# 3. Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df_clust['Cluster'] = kmeans.fit_predict(scaled_data)
```
