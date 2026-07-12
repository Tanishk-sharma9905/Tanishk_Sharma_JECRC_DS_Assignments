import os
import re
import pandas as pd
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import datetime
import tempfile

# Feature 2: Business KPI Dashboard Calculations
def profile_business_kpis(df):
    """
    Scans column names fuzzy-matching business terms.
    Returns a dictionary of calculated KPIs.
    """
    kpis = {}
    
    # Clean up column names for matching
    cols = {col.lower().replace("_", "").replace(" ", ""): col for col in df.columns}
    
    # Revenue / Sales
    rev_col = None
    for term in ["revenue", "sales", "turnover", "grandtotal", "totalamount"]:
        if term in cols:
            rev_col = cols[term]
            break
            
    # Profit
    profit_col = None
    for term in ["profit", "netincome", "gain", "margin"]:
        if term in cols:
            profit_col = cols[term]
            break
            
    # Quantity / Orders
    qty_col = None
    for term in ["quantity", "qty", "units", "items"]:
        if term in cols:
            qty_col = cols[term]
            break
            
    # Ratings
    rating_col = None
    for term in ["rating", "score", "reviewscore", "ratingvalue"]:
        if term in cols:
            rating_col = cols[term]
            break

    # Order ID / Transactions
    id_col = None
    for term in ["orderid", "transactionid", "invoiceid", "id", "custid"]:
        if term in cols:
            id_col = cols[term]
            break

    # Price / Cost
    price_col = None
    for term in ["price", "cost", "unitprice", "rate"]:
        if term in cols:
            price_col = cols[term]
            break

    # Compute values
    if rev_col and pd.api.types.is_numeric_dtype(df[rev_col]):
        total_rev = df[rev_col].sum()
        kpis["Total Revenue"] = f"${total_rev:,.2f}"
        kpis["Average Sales"] = f"${df[rev_col].mean():,.2f}"
        kpis["Maximum Sale"] = f"${df[rev_col].max():,.2f}"
        kpis["Minimum Sale"] = f"${df[rev_col].min():,.2f}"
        
        if id_col:
            order_count = df[id_col].nunique()
            kpis["Order Count"] = f"{order_count:,}"
            kpis["Average Order Value"] = f"${total_rev / max(1, order_count):,.2f}"
            
    if profit_col and pd.api.types.is_numeric_dtype(df[profit_col]):
        total_profit = df[profit_col].sum()
        kpis["Total Profit"] = f"${total_profit:,.2f}"
        kpis["Average Profit"] = f"${df[profit_col].mean():,.2f}"
        
        # Calculate Margin if Revenue is available
        if rev_col and pd.api.types.is_numeric_dtype(df[rev_col]):
            total_rev = df[rev_col].sum()
            margin = (total_profit / total_rev * 100) if total_rev != 0 else 0
            kpis["Profit Margin"] = f"{margin:.2f}%"
            
    if rating_col and pd.api.types.is_numeric_dtype(df[rating_col]):
        kpis["Average Customer Rating"] = f"{df[rating_col].mean():.2f} / 5.0"

    return kpis

# Feature 8: Dataset Health Score
def get_health_score(df):
    """
    Calculates a data quality score out of 100 based on
    missing values, duplicates, outliers, types, and dimensions.
    """
    score = 100
    recs = []
    
    # 1. Missing Values Deduction
    total_cells = df.size
    if total_cells > 0:
        missing_cells = df.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells
        missing_deduction = round(missing_ratio * 40, 2)
        score -= missing_deduction
        if missing_deduction > 0:
            recs.append(f"Impute or remove missing cells ({missing_cells:,} cells, {missing_ratio*100:.2f}% of dataset).")
            
    # 2. Duplicates Deduction
    duplicate_rows = df.duplicated().sum()
    if len(df) > 0 and duplicate_rows > 0:
        dup_ratio = duplicate_rows / len(df)
        dup_deduction = round(dup_ratio * 20, 2)
        score -= dup_deduction
        recs.append(f"Remove or inspect {duplicate_rows:,} duplicate records ({dup_ratio*100:.2f}% of dataset).")

    # 3. Outliers Deduction
    num_cols = df.select_dtypes(include='number').columns.tolist()
    total_outliers = 0
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        total_outliers += len(outliers)
        
    if len(df) > 0 and total_outliers > 0:
        outlier_ratio = total_outliers / (len(df) * max(1, len(num_cols)))
        outlier_deduction = min(20, round(outlier_ratio * 30, 2))
        score -= outlier_deduction
        if outlier_deduction > 5:
            recs.append("Apply Z-score or IQR thresholds to filter statistical outliers in numeric columns.")

    # 4. Type Completeness
    object_cols = df.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        try:
            # Check if dates are stored as string
            sample = df[col].dropna().head(5)
            if not sample.empty:
                try:
                    converted = pd.to_datetime(sample, format='mixed')
                    if converted.dtype.kind == 'M':
                        score -= 2
                        recs.append(f"Convert column '{col}' from string object to native DateTime type.")
                except Exception:
                    pass
        except Exception:
            pass

    score = max(0, min(100, round(score)))
    if score >= 90:
        grade = "Excellent"
    elif score >= 75:
        grade = "Good"
    elif score >= 50:
        grade = "Fair"
    else:
        grade = "Poor"
        
    return {
        "score": score,
        "grade": grade,
        "recommendations": recs if recs else ["Dataset is clean and ready for analytical modeling."]
    }

# Feature 9: Outlier Detection
def detect_outliers(df):
    """
    Computes outliers using IQR and standard Z-score method.
    Returns details for each numeric column.
    """
    num_cols = df.select_dtypes(include='number').columns
    outlier_info = {}
    
    for col in num_cols:
        # IQR Method
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        # Z-score Method
        col_mean = df[col].mean()
        col_std = df[col].std()
        if col_std > 0:
            z_scores = (df[col] - col_mean) / col_std
            z_outliers = df[np.abs(z_scores) > 3]
        else:
            z_outliers = pd.DataFrame()

        if len(iqr_outliers) > 0 or len(z_outliers) > 0:
            outlier_info[col] = {
                "iqr_count": len(iqr_outliers),
                "iqr_pct": round(len(iqr_outliers) / len(df) * 100, 2),
                "z_count": len(z_outliers),
                "z_pct": round(len(z_outliers) / len(df) * 100, 2),
                "bounds": (round(lower_bound, 2), round(upper_bound, 2))
            }
            
    return outlier_info

# Feature 10: Statistical Expansion
def get_expanded_statistics(df):
    """
    Extends profiling stats with Variance, Covariance, Range, CoV, IQR, Percentiles.
    """
    num_cols = df.select_dtypes(include='number').columns
    stats = {}
    
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        col_mean = df[col].mean()
        col_std = df[col].std()
        q25 = df[col].quantile(0.25)
        q50 = df[col].quantile(0.50)
        q75 = df[col].quantile(0.75)
        
        stats[col] = {
            "Variance": df[col].var(),
            "Coefficient of Variation": (col_std / col_mean) if col_mean != 0 else 0,
            "Range": col_max - col_min,
            "IQR": q75 - q25,
            "25th Percentile": q25,
            "50th Percentile": q50,
            "75th Percentile": q75
        }
    return stats

# Feature 5: Automatic Visualization Recommendations
def recommend_visualizations(df):
    """
    Analyzes dataset columns and recommends useful chart combinations.
    """
    cols = {col.lower().replace("_", "").replace(" ", ""): col for col in df.columns}
    recs = []
    
    # Check for timeseries
    date_col = None
    for term in ["date", "time", "timestamp", "year", "month", "day"]:
        if term in cols:
            date_col = cols[term]
            break
            
    # Check for categoricals
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Check for numericals
    num_cols = df.select_dtypes(include='number').columns.tolist()
    
    # Generate recommendations
    if date_col and num_cols:
        recs.append(f"📈 Monthly Trend: Plotting '{num_cols[0]}' over '{date_col}' to evaluate business growth.")
        
    if cat_cols and num_cols:
        recs.append(f"📊 Categorical Distribution: Bar chart showing '{num_cols[0]}' aggregated by '{cat_cols[0]}'.")
        if len(cat_cols) > 1:
            recs.append(f"🍕 Contribution Share: Pie chart showing share contribution of '{cat_cols[1]}'.")
            
    if len(num_cols) >= 2:
        recs.append(f"🔵 Scatter Matrix: Correlation scatter plot comparing '{num_cols[0]}' against '{num_cols[1]}'.")
        recs.append("🔥 Correlation Heatmap: Evaluating linear relationships across all numeric dimensions.")
        
    if num_cols:
        recs.append(f"🔔 Density Profile: Histogram showing distribution and skewness of '{num_cols[0]}'.")

    return recs[:5]

# Feature 12: Suggested Questions
def suggest_questions(df):
    """
    Suggests business analytics questions tailored to columns.
    """
    cols = {col.lower().replace("_", "").replace(" ", ""): col for col in df.columns}
    questions = []
    
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()
    
    if num_cols:
        questions.append(f"Show total values and statistics of {num_cols[0]}")
        if cat_cols:
            questions.append(f"Compare average {num_cols[0]} across different {cat_cols[0]} categories")
            
    # Time analysis
    date_col = None
    for term in ["date", "time", "timestamp", "year", "month"]:
        if term in cols:
            date_col = cols[term]
            break
    if date_col and num_cols:
        questions.append(f"Show monthly trend of {num_cols[0]}")
        
    if len(num_cols) >= 2:
        questions.append(f"Show scatter relationship between {num_cols[0]} and {num_cols[1]}")
        
    # Anomaly / Outliers
    if num_cols:
        questions.append(f"Detect statistical outliers in {num_cols[0]}")
        
    # ML / Clustering
    if len(num_cols) >= 2:
        questions.append("Perform customer segmentation or item clustering")

    return questions[:6]

# Feature 16: Basic Forecasting Helper
def run_forecast(df, value_col, time_col):
    """
    Performs moving average and linear regression forecasting on time series.
    Returns Plotly figure.
    """
    df_clean = df[[time_col, value_col]].dropna().copy()
    df_clean[time_col] = pd.to_datetime(df_clean[time_col])
    df_clean = df_clean.sort_values(time_col)
    
    # Resample to monthly to keep it clean
    df_clean.set_index(time_col, inplace=True)
    df_m = df_clean.resample('ME').mean()
    
    # Moving Average
    df_m['Moving Average (3M)'] = df_m[value_col].rolling(window=3, min_periods=1).mean()
    
    # Linear Regression
    X = np.arange(len(df_m)).reshape(-1, 1)
    y = df_m[value_col].values
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    df_m['Trendline (Linear)'] = model.predict(X)
    
    # Forecast 3 months out
    future_X = np.arange(len(df_m), len(df_m)+3).reshape(-1, 1)
    future_preds = model.predict(future_X)
    
    future_dates = [df_m.index[-1] + pd.DateOffset(months=i) for i in range(1, 4)]
    df_future = pd.DataFrame(future_preds, index=future_dates, columns=['Trendline (Linear)'])
    
    df_all = pd.concat([df_m, df_future])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_m.index, y=df_m[value_col], name='Actual', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df_m.index, y=df_m['Moving Average (3M)'], name='3-Month Moving Average', mode='lines'))
    fig.add_trace(go.Scatter(x=df_all.index, y=df_all['Trendline (Linear)'], name='Linear Trend & Forecast', mode='lines', line=dict(dash='dash')))
    
    fig.update_layout(title=f"3-Month Forecast: {value_col} over Time", xaxis_title=time_col, yaxis_title=value_col)
    return fig

# Feature 17: Anomaly Detection
def detect_anomalies(df):
    """
    Finds unexpected spikes or dips in numerical columns.
    """
    num_cols = df.select_dtypes(include='number').columns
    anomalies = {}
    
    for col in num_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        if col_std > 0:
            z_scores = (df[col] - col_mean) / col_std
            spikes = df[z_scores > 3]
            dips = df[z_scores < -3]
            if len(spikes) > 0 or len(dips) > 0:
                anomalies[col] = {
                    "spikes_count": len(spikes),
                    "dips_count": len(dips),
                    "highest_anomaly": df[col].max(),
                    "lowest_anomaly": df[col].min()
                }
    return anomalies

# Feature: KMeans Cohort Segmentation
def run_cohort_segmentation(df, n_clusters=3):
    """
    Segments dataset using KMeans on numeric columns.
    Returns a Plotly scatter/parallel-coordinates figure and a brief summary of clusters.
    """
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) < 2:
        return None, "Not enough numerical columns to perform cohort segmentation."
    
    # Select top numeric columns (up to 3 for visualization)
    cols_to_use = num_cols[:3]
    
    # Drop missing values for clustering
    df_clust = df[cols_to_use].dropna().copy()
    if len(df_clust) < n_clusters * 2:
        return None, "Insufficient data points after dropping missing values."
        
    # Normalize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clust)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df_clust['Cluster'] = kmeans.fit_predict(scaled_data)
    df_clust['Cluster'] = df_clust['Cluster'].astype(str)
    
    # Visualize
    if len(cols_to_use) >= 3:
        fig = px.scatter_3d(
            df_clust, x=cols_to_use[0], y=cols_to_use[1], z=cols_to_use[2],
            color='Cluster', title=f"3D Cohort Segmentation using KMeans (k={n_clusters})",
            opacity=0.8
        )
    else:
        fig = px.scatter(
            df_clust, x=cols_to_use[0], y=cols_to_use[1],
            color='Cluster', title=f"Cohort Segmentation using KMeans (k={n_clusters})"
        )
    
    # Summarize clusters
    cluster_means = df_clust.groupby('Cluster').mean()
    summary_parts = []
    for cluster, row in cluster_means.iterrows():
        metrics = ", ".join([f"{col}: {row[col]:.2f}" for col in cols_to_use])
        summary_parts.append(f"Cluster {cluster}: Mean values -> {metrics}")
    summary_text = "\n".join(summary_parts)
    
    return fig, summary_text

# Feature 7: PDF Report Exporter
class BusinessReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(30, 144, 255)
        self.cell(0, 10, 'Executive Business Report', border=False, new_x="LMARGIN", new_y="NEXT", align='L')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf_report(df, file_name):
    """
    Compiles PDF report with Overview, KPI statistics, Data Quality, and stats.
    """
    pdf = BusinessReportPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    
    # Title Block
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, f"Analysis Target: {file_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 8, f"Date Compiled: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # 1. Executive Summary
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "1. Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    summary_text = (
        f"This report presents an evaluation of the dataset '{file_name}'. "
        f"The data contains {df.shape[0]:,} observations structured across {df.shape[1]:,} columns. "
        f"A total of {df.duplicated().sum():,} duplicate rows and {df.isnull().sum().sum():,} missing data cells were detected during initial profiling. "
        f"The overall structures indicate the dataset is suited for operational reporting and analytics modeling."
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(5)
    
    # 2. Business KPIs
    kpis = profile_business_kpis(df)
    if kpis:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "2. Key Business Metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        for k, v in kpis.items():
            pdf.cell(80, 6, f"- {k}:", border=False)
            pdf.cell(0, 6, f" {v}", border=False, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
    # 3. Data Quality
    health = get_health_score(df)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "3. Data Quality & Health Assessment", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Dataset Health Score: {health['score']}/100 ({health['grade']})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.cell(0, 6, "Key Improvements Recommended:", new_x="LMARGIN", new_y="NEXT")
    for rec in health["recommendations"]:
        pdf.multi_cell(0, 5, f"- {rec}")
    pdf.ln(5)
    
    # 4. Statistical Highlights
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "4. Numeric Column Highlights", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    num_cols = df.select_dtypes(include='number').columns[:5]
    for col in num_cols:
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 6, f"Column: {col}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        desc = df[col].describe()
        pdf.cell(40, 5, f"Mean: {desc['mean']:.2f}")
        pdf.cell(40, 5, f"Min: {desc['min']:.2f}")
        pdf.cell(40, 5, f"Max: {desc['max']:.2f}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        
    # Save PDF
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, f"{file_name}_report.pdf")
    pdf.output(pdf_path)
    return pdf_path
