import streamlit as st
import pandas as pd
import os
import shutil
import sqlite3
import zipfile
import tempfile
import datetime
import time
import re
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from src.agent import DataScienceAgent, AVAILABLE_MODELS, DEFAULT_MODEL
from src.business_analyst import (
    profile_business_kpis,
    get_health_score,
    detect_outliers,
    get_expanded_statistics,
    recommend_visualizations,
    suggest_questions,
    detect_anomalies,
    run_forecast,
    generate_pdf_report
)

load_dotenv()

st.set_page_config(page_title="Autonomous Data Science Co-Pilot", layout="wide")

# Helper to extract a markdown section by header name
def get_section(text, header_name):
    keywords = {
        "Executive Summary": [r"executive\s*summary", r"overview", r"summary"],
        "Key Insights": [r"key\s*insights", r"insights", r"findings"],
        "Business Recommendations": [r"business\s*recommendations", r"recommendations", r"actions", r"recommendation"],
        "Technical Summary": [r"technical\s*summary", r"technical\s*details", r"technical"]
    }
    patterns = keywords.get(header_name, [header_name])
    for pat in patterns:
        pattern = rf"(#+\s*[^#\n]*{pat}[^#\n]*\n.*?)(?=(?:\n#+\s*)|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""

# Sidebar - User isolation & Project Management
st.sidebar.title("👤 User Isolation")
user_id = st.sidebar.text_input("User ID / Username:", value="default_user", help="Isolates your projects and conversation history from other users.")
user_id = re.sub(r'[^A-Za-z0-9_-]', '', user_id.strip()) if user_id.strip() else "default_user"

st.sidebar.title("📁 Project Management")
base_workspace = os.path.abspath(os.path.join("workspace", user_id))
os.makedirs(base_workspace, exist_ok=True)

existing_projects = [d for d in os.listdir(base_workspace) if os.path.isdir(os.path.join(base_workspace, d))]
if not existing_projects:
    existing_projects = ["default_project"]

project_name = st.sidebar.selectbox("Select Project Workspace", existing_projects, index=0)
project_name = re.sub(r'[^A-Za-z0-9_-]', '', project_name.strip()) if project_name else "default_project"

st.sidebar.markdown("### ➕ Create New Project")
new_project_name = st.sidebar.text_input("New Project Name:")
if st.sidebar.button("Create Project"):
    new_project_name_clean = re.sub(r'[^A-Za-z0-9_-]', '', new_project_name.strip())
    if new_project_name_clean:
        new_project_dir = os.path.join(base_workspace, new_project_name_clean)
        if os.path.abspath(new_project_dir).startswith(os.path.abspath(base_workspace)):
            os.makedirs(new_project_dir, exist_ok=True)
            os.makedirs(os.path.join(new_project_dir, "uploads"), exist_ok=True)
            st.sidebar.success(f"Project '{new_project_name_clean}' created successfully!")
            st.rerun()
        else:
            st.sidebar.error("Invalid project path.")
    else:
        st.sidebar.error("Project name cannot be empty or contain invalid characters.")

project_dir = os.path.join(base_workspace, project_name)
if not os.path.abspath(project_dir).startswith(os.path.abspath(base_workspace)):
    st.error("Invalid project workspace location.")
    st.stop()

uploads_dir = os.path.join(project_dir, "uploads")
os.makedirs(uploads_dir, exist_ok=True)

st.sidebar.markdown("---")

# Sidebar - Model Configuration
st.sidebar.title("🤖 Model Configuration")
model_choice = st.sidebar.selectbox(
    "Choose Gemini Model", 
    list(AVAILABLE_MODELS.keys()),
    index=0
)
model_name = AVAILABLE_MODELS[model_choice]

# Sidebar - Authentication
st.sidebar.markdown("---")
st.sidebar.title("🔑 Authentication")
show_key = st.sidebar.checkbox("Show API Key", value=False)
user_api_key = st.sidebar.text_input(
    "Enter your Gemini API Key", 
    value=os.getenv("GEMINI_API_KEY", ""),
    type="default" if show_key else "password", 
    autocomplete="off",
    help="Enter your Gemini API Key. If GEMINI_API_KEY is defined in .env, it will be used as the default prefilled value."
)

if user_api_key.strip():
    st.session_state.gemini_api_key = user_api_key.strip()
else:
    st.session_state.gemini_api_key = ""
st.session_state.gemini_model = model_name

with st.sidebar.expander("Don't have a Gemini API Key?"):
    st.markdown("""
    1. Visit [Google AI Studio](https://aistudio.google.com/apikey).
    2. Sign in with your Google account.
    3. Click **"Create API Key"**.
    4. Copy the generated API key.
    5. Paste it in the input field above.
    """)
    st.markdown('<a href="https://aistudio.google.com/apikey" target="_blank"><button style="border-radius:4px;background-color:#1E90FF;color:white;padding:8px 16px;border:none;cursor:pointer;">Generate Free Gemini API Key</button></a>', unsafe_allow_html=True)

# Helper to check API key (Strictly rely on the key entered by the user)
api_key = st.session_state.get("gemini_api_key", "").strip()

# Debugging Mode
st.sidebar.markdown("---")
enable_debug = st.sidebar.checkbox("Enable Debugging Mode", value=False)

if enable_debug:
    st.sidebar.info(f"🔍 Model Debug: Label='{model_choice}' -> Actual ID Passed='{model_name}'")
    api_source = "None"
    masked_key = "None"
    if api_key:
        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        if env_key and api_key == env_key:
            api_source = "Environment Variable"
        else:
            api_source = "User Input"
            
        if len(api_key) > 8:
            masked_key = api_key[:6] + "*" * (len(api_key) - 10) + api_key[-4:]
        else:
            masked_key = "*" * len(api_key)
            
    st.sidebar.write(f"**Active Model**: `{model_name}`")
    st.sidebar.write(f"**API Source**: `{api_source}`")
    st.sidebar.write(f"**API Key (Masked)**: `{masked_key}`")


# Initialize session state for report content
current_project_key = f"{user_id}/{project_name}"
if "current_project" not in st.session_state or st.session_state.get("current_project") != current_project_key:
    st.session_state.current_project = current_project_key
    st.session_state.report_content = ""

# Main Content Header
st.title(f"🚀 Autonomous Data Science Co-Pilot - {project_name}")
st.markdown("##### Upload your dataset and query it in plain English. The agent handles inspection, preprocessing, visual mapping, and code healing autonomously.")

# Issue 1 & 2: Show only input key error if empty
if not api_key:
    st.warning("Please enter your Gemini API Key before running an analysis.")
    st.stop()

uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx", "xls", "json", "parquet", "tsv", "sqlite", "zip"])

@st.cache_data
def get_sqlite_tables(file_path):
    conn = sqlite3.connect(file_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    conn.close()
    return tables['name'].tolist() if not tables.empty else []

@st.cache_data
def load_data(file_path, table_name=None):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif file_path.endswith(".tsv"):
        return pd.read_csv(file_path, sep="\t")
    elif file_path.endswith(".sqlite"):
        conn = sqlite3.connect(file_path)
        if not table_name:
            tables = get_sqlite_tables(file_path)
            table_name = tables[0] if tables else None
        if not table_name:
            conn.close()
            raise Exception("No tables found in SQLite database.")
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    elif file_path.endswith(".zip"):
        with tempfile.TemporaryDirectory() as extract_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            data_files = []
            for root, dirs, files in os.walk(extract_dir):
                for f in files:
                    if f.endswith((".csv", ".parquet", ".json", ".xlsx", ".xls", ".tsv")):
                        data_files.append(os.path.join(root, f))
            if not data_files:
                raise Exception("No supported data file found in ZIP.")
            selected_extracted = data_files[0]
            selected_name = os.path.basename(selected_extracted)
            new_target = os.path.join(uploads_dir, selected_name)
            shutil.copy(selected_extracted, new_target)
            if len(data_files) > 1:
                ignored = [os.path.basename(x) for x in data_files[1:]]
                st.session_state.zip_warning = f"Warning: Multiple data files found in ZIP. Loaded '{selected_name}'. Ignored: {', '.join(ignored)}"
            else:
                st.session_state.zip_warning = None
            return load_data(new_target)
        raise Exception("No supported data file found in ZIP.")
    else:
        raise Exception("Unsupported file type")

@st.cache_data
def profile_dataset(df, file_name):
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    
    dt_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_cols.append(col)
        elif df[col].dtype == 'object':
            try:
                sample = df[col].dropna().head(5)
                if not sample.empty:
                    try:
                        converted = pd.to_datetime(sample, format='mixed')
                        if converted.dtype.kind == 'M':
                            dt_cols.append(col)
                    except Exception:
                        pass
            except Exception:
                pass
                
    max_cols_for_detail = 15
    truncated_detail = len(df.columns) > max_cols_for_detail
    cols_to_profile = df.columns[:max_cols_for_detail].tolist()
    
    summary = f"File name: {file_name}\n"
    summary += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
    summary += f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
    summary += f"Duplicate Rows: {df.duplicated().sum()}\n"
    
    if truncated_detail:
        summary += f"\n⚠️ NOTE: Dataset has {len(df.columns)} columns. Detailed profiling is capped at the first {max_cols_for_detail} columns to optimize prompt size.\n"
        
    summary += f"Missing Values (capped):\n{df[cols_to_profile].isnull().sum().to_string()}\n"
    summary += f"Numeric Columns ({len(num_cols)}): {', '.join(num_cols[:max_cols_for_detail])}{' ... (truncated)' if len(num_cols) > max_cols_for_detail else ''}\n"
    summary += f"Categorical Columns ({len(cat_cols)}): {', '.join(cat_cols[:max_cols_for_detail])}{' ... (truncated)' if len(cat_cols) > max_cols_for_detail else ''}\n"
    summary += f"Datetime Columns ({len(dt_cols)}): {', '.join(dt_cols[:max_cols_for_detail])}{' ... (truncated)' if len(dt_cols) > max_cols_for_detail else ''}\n"
    summary += f"Unique Values (first 10 cols):\n{df.iloc[:, :10].nunique().to_string()}\n"
    
    profiled_num_cols = [c for c in num_cols if c in cols_to_profile]
    if profiled_num_cols:
        summary += f"\nStatistical Summary (capped):\n{df[profiled_num_cols].describe().to_string()}\n"
        try:
            summary += f"\nSkewness (capped):\n{df[profiled_num_cols].skew().to_string()}\n"
            summary += f"Kurtosis (capped):\n{df[profiled_num_cols].kurt().to_string()}\n"
            if len(profiled_num_cols) > 1 and len(profiled_num_cols) <= 15:
                summary += f"\nCorrelation Matrix (capped):\n{df[profiled_num_cols].corr().to_string()}\n"
        except Exception:
            pass

    return summary

if uploaded_file:
    # Size limit guardrail (50MB)
    MAX_FILE_SIZE_MB = 50
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File size exceeds the limit of {MAX_FILE_SIZE_MB}MB. Please upload a smaller dataset.")
        st.stop()
        
    file_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"File '{uploaded_file.name}' successfully uploaded to {project_name}/uploads!")
    
    try:
        table_name = None
        if uploaded_file.name.endswith(".sqlite"):
            tables = get_sqlite_tables(file_path)
            if len(tables) > 1:
                table_name = st.selectbox("Select Table to Analyze", tables)
            elif len(tables) == 1:
                table_name = tables[0]
                
        df = load_data(file_path, table_name)
        if st.session_state.get("zip_warning"):
            st.warning(st.session_state.zip_warning)
            
        # Row count limit guardrail (500k rows)
        MAX_ROW_COUNT = 500000
        if len(df) > MAX_ROW_COUNT:
            st.error(f"Dataset has {len(df):,} rows, which exceeds the limit of {MAX_ROW_COUNT:,} rows. Please upload a smaller dataset.")
            st.stop()
            
        # De-duplicate column names immediately
        if not df.columns.is_unique:
            new_cols = []
            col_counts = {}
            for col in df.columns:
                if col in col_counts:
                    col_counts[col] += 1
                    new_cols.append(f"{col}_{col_counts[col]}")
                else:
                    col_counts[col] = 0
                    new_cols.append(col)
            df.columns = new_cols
            
        summary = profile_dataset(df, uploaded_file.name)
        
        # New Feature Tabs (Backwards compatible, preserving profiling logic)
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Quality & Profiling", 
            "⚠️ Outliers & Anomalies", 
            "⚡ Business Dashboard", 
            "📥 Report & Downloads"
        ])
        
        # Tab 1: Profiling, Quality, KPI Cards, and Health score
        with tab1:
            # Feature 8: Health Score
            health = get_health_score(df)
            st.markdown(f"### 🩺 Dataset Health Score: **{health['score']}/100** ({health['grade']})")
            
            # Feature 1: Executive Summary KPIs
            st.markdown("### 📋 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{df.shape[0]:,}")
            col2.metric("Total Columns", f"{df.shape[1]:,}")
            col3.metric("Memory Footprint", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            col4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
            
            # Numeric, Categorical and Date columns
            num_cols = df.select_dtypes(include='number').columns.tolist()
            cat_cols = df.select_dtypes(exclude='number').columns.tolist()
            
            # DateTime column detection
            dt_cols = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    dt_cols.append(col)
                elif df[col].dtype == 'object':
                    try:
                        sample = df[col].dropna().head(5)
                        if not sample.empty:
                            try:
                                converted = pd.to_datetime(sample, format='mixed')
                                if converted.dtype.kind == 'M':
                                    dt_cols.append(col)
                            except Exception:
                                pass
                    except Exception:
                        pass

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Numeric Columns", len(num_cols))
            col6.metric("Categorical Columns", len(cat_cols))
            col7.metric("Date Columns", len(dt_cols))
            col8.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            
            # Feature 1: Executive summary paragraph
            st.markdown("#### **Executive Overview**")
            st.markdown(
                f"- Dataset contains **{df.shape[0]:,}** records across **{df.shape[1]:,}** columns.\n"
                f"- **{df.isnull().sum().sum():,}** missing values detected.\n"
                f"- **{df.duplicated().sum():,}** duplicate records found.\n"
                f"- The dataset health is graded as **{health['grade']}** and suitable for business analytics."
            )
            
            # Feature 2: Business KPI Dashboard
            kpis = profile_business_kpis(df)
            if kpis:
                st.markdown("### 💼 Key Performance Indicators (KPIs)")
                kpi_cols = st.columns(min(len(kpis), 4))
                for idx, (k, v) in enumerate(kpis.items()):
                    kpi_cols[idx % 4].metric(k, v)

            # Quality report recommendations
            st.markdown("#### **Quality Recommendations**")
            for rec in health["recommendations"]:
                st.markdown(f"- 💡 {rec}")

            # Missing values table
            st.markdown("### 🔍 Column Types & Data Quality")
            missing_counts = df.isnull().sum()
            missing_pct = (missing_counts / len(df) * 100).round(2)
            quality_df = pd.DataFrame({
                "Data Type": df.dtypes.astype(str),
                "Missing Count": missing_counts,
                "Missing Percentage": missing_pct.astype(str) + "%",
                "Unique Values Count": df.nunique()
            })
            st.dataframe(quality_df, use_container_width=True)

            # Feature 10: Statistical Expansion
            if num_cols:
                st.markdown("### 📈 Statistical Highlights")
                st.dataframe(df[num_cols].describe().T, use_container_width=True)
                
                with st.expander("Detailed Skewness & Kurtosis Statistics"):
                    skew_kurt_df = pd.DataFrame({
                        "Skewness": df[num_cols].skew(),
                        "Kurtosis": df[num_cols].kurt()
                    })
                    st.dataframe(skew_kurt_df, use_container_width=True)

                # Feature 10: Extended statistics
                with st.expander("Extended Statistics (Variance, Coefficient of Variation, Range, IQR, Percentiles)"):
                    ext_stats = get_expanded_statistics(df)
                    st.dataframe(pd.DataFrame(ext_stats).T, use_container_width=True)

                if len(num_cols) > 1:
                    with st.expander("Top Correlations Matrix"):
                        st.dataframe(df[num_cols].corr(), use_container_width=True)
                        
            # Feature 5: Automatic Visualization Recommendations
            recommendations = recommend_visualizations(df)
            if recommendations:
                st.markdown("### 🎨 Recommended Visualizations")
                for r in recommendations:
                    st.markdown(r)
                        
            st.markdown("### 📄 Preview (First 100 Rows)")
            st.dataframe(df.head(100), use_container_width=True)

        # Tab 2: Outliers & Anomalies
        with tab2:
            st.markdown("### ⚠️ Outliers & Anomaly Detection")
            
            # Feature 9: Outlier Detection
            outliers = detect_outliers(df)
            if outliers:
                st.markdown("#### **Outliers Detected (IQR & Z-score)**")
                st.dataframe(pd.DataFrame(outliers).T, use_container_width=True)
                
                # Show outlier plot for first numerical column
                col = num_cols[0]
                fig = px.box(df, y=col, title=f"Outlier Distribution in {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No statistical outliers detected.")
                
            # Feature 17: Anomaly Detection Spikes/Dips
            anomalies = detect_anomalies(df)
            if anomalies:
                st.markdown("#### **Data Spikes and Dips (Anomalies)**")
                st.dataframe(pd.DataFrame(anomalies).T, use_container_width=True)
            else:
                st.success("No anomalous spikes or dips found in numerical fields.")
                
            st.markdown("---")
            st.markdown("### 🧬 Interactive Customer Cohort Segmentation")
            if len(num_cols) >= 2:
                n_clusters_input = st.slider("Select Number of Clusters (k)", min_value=2, max_value=8, value=3)
                if st.button("Run KMeans Cohort Analysis"):
                    from src.business_analyst import run_cohort_segmentation
                    fig_clust, summary_clust = run_cohort_segmentation(df, n_clusters=n_clusters_input)
                    if fig_clust is not None:
                        st.plotly_chart(fig_clust, use_container_width=True)
                        st.text("Cohort Summary:")
                        st.code(summary_clust)
                    else:
                        st.error(summary_clust)
            else:
                st.info("Cohort segmentation requires at least 2 numerical columns.")

        # Tab 3: Business Dashboard Generation
        with tab3:
            st.markdown("### ⚡ Auto-Generated Business Dashboard")
            if st.button("Generate Dashboard"):
                # Feature 6: Dashboard logic
                cols = {col.lower().replace("_", "").replace(" ", ""): col for col in df.columns}
                
                # Metric Cards
                kpis = profile_business_kpis(df)
                if kpis:
                    dashboard_cols = st.columns(min(len(kpis), 4))
                    for idx, (k, v) in enumerate(kpis.items()):
                        dashboard_cols[idx % 4].metric(k, v)
                
                # Try Region chart
                region_col = None
                for term in ["region", "country", "state", "city", "location", "territory"]:
                    if term in cols:
                        region_col = cols[term]
                        break
                
                # Revenue / Sales
                val_col = None
                for term in ["revenue", "sales", "turnover", "grandtotal"]:
                    if term in cols:
                        val_col = cols[term]
                        break
                        
                # Date
                date_col = None
                for term in ["date", "time", "timestamp", "year", "month"]:
                    if term in cols:
                        date_col = cols[term]
                        break

                # Ratings
                rating_col = None
                for term in ["rating", "score"]:
                    if term in cols:
                        rating_col = cols[term]
                        break

                col1, col2 = st.columns(2)
                
                # Sales by Region chart
                if region_col and val_col:
                    fig_region = px.bar(
                        df.groupby(region_col)[val_col].sum().reset_index(),
                        x=region_col, y=val_col,
                        title=f"Total Sales by {region_col}"
                    )
                    col1.plotly_chart(fig_region, use_container_width=True)
                    
                # Monthly Trend chart
                if date_col and val_col:
                    df_trend = df[[date_col, val_col]].dropna().copy()
                    df_trend[date_col] = pd.to_datetime(df_trend[date_col])
                    df_trend = df_trend.groupby(df_trend[date_col].dt.to_period("M"))[val_col].sum().reset_index()
                    df_trend[date_col] = df_trend[date_col].astype(str)
                    fig_trend = px.line(df_trend, x=date_col, y=val_col, title="Monthly Sales Trend", markers=True)
                    col2.plotly_chart(fig_trend, use_container_width=True)
                    
                # Distribution of Customer Ratings
                if rating_col:
                    fig_rating = px.histogram(df, x=rating_col, nbins=10, title="Customer Rating Distribution")
                    col1.plotly_chart(fig_rating, use_container_width=True)
                    
                # Correlation Heatmap
                if len(num_cols) > 1:
                    fig_corr = px.imshow(df[num_cols].corr(), text_auto=True, title="Correlation Matrix Heatmap")
                    col2.plotly_chart(fig_corr, use_container_width=True)
                    
            st.markdown("---")
            st.markdown("### 📈 Interactive Time-Series Forecasting")
            if dt_cols and num_cols:
                col_t1, col_t2 = st.columns(2)
                time_col_selection = col_t1.selectbox("Select Date/Time Column", dt_cols, key="fc_time")
                value_col_selection = col_t2.selectbox("Select Value Column to Forecast", num_cols, key="fc_val")
                if st.button("Generate 3-Month Forecast"):
                    try:
                        df_fc = df.copy()
                        df_fc[time_col_selection] = pd.to_datetime(df_fc[time_col_selection], errors='coerce')
                        fig_forecast = run_forecast(df_fc, value_col_selection, time_col_selection)
                        st.plotly_chart(fig_forecast, use_container_width=True)
                    except Exception as ex:
                        st.error(f"Forecasting failed: {ex}")
            else:
                st.info("Forecasting requires at least one date/time column and one numeric column.")

        # Tab 4: Exporter, PDF reports, cleaned dataset download
        with tab4:
            st.markdown("### 📥 Reports & File Exporter")
            
            # PDF Business Report
            if st.button("Generate Executive PDF Report"):
                pdf_path = generate_pdf_report(df, uploaded_file.name)
                with open(pdf_path, "rb") as f:
                    st.download_button("Download Executive Business Report (PDF)", f, file_name=f"{uploaded_file.name}_business_report.pdf")
            
            # Markdown Report Export
            if st.session_state.report_content:
                st.download_button("Export Session Report (Markdown)", st.session_state.report_content, file_name=f"{project_name}_report.md")
                
            # Cleaned Dataset Export
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Cleaned Dataset (CSV)", csv_data, file_name=f"{uploaded_file.name}_cleaned.csv")
            
    except Exception as e:
        st.error(f"Could not read the uploaded file for summary: {str(e)}")
        summary = f"File name: {uploaded_file.name}. Could not parse preview."

    # Feature 12: Clickable Suggested Prompts
    st.markdown("---")
    questions = suggest_questions(df)
    st.markdown("💡 **Suggested Analysis Prompts:**")
    q_cols = st.columns(3)
    for idx, q in enumerate(questions):
        if q_cols[idx % 3].button(q, key=f"q_btn_{idx}"):
            st.session_state.prompt_input = q
            st.rerun()

    # User Input Field
    user_question = st.text_input("💡 What would you like to know or analyze?", value=st.session_state.get("prompt_input", ""))
    
    if st.button("Analyze ⚡") and user_question:
        # Issue 6: Agent Initialization Verification
        if not api_key:
            st.error("Please enter your Gemini API Key.")
        elif not uploaded_file:
            st.error("Please upload a dataset first.")
        else:
            try:
                start_time = time.time()
                with st.status("Analyzing...", expanded=True) as status:
                    st.write("Initializing Agent...")
                    @st.cache_resource
                    def get_cached_rag():
                        from src.rag import RAGPipeline
                        return RAGPipeline()
                        
                    agent = DataScienceAgent(
                        api_key=api_key, 
                        model_name=st.session_state.get("gemini_model", DEFAULT_MODEL), 
                        user_id=user_id, 
                        rag_pipeline=get_cached_rag()
                    )
                    
                    st.write("Generating Plan and Code...")
                    result = agent.process_request(project_name, summary, file_path, user_question)
                    
                    st.write("Executing and Validating...")
                    exec_result = result["execution_result"]
                    
                    if result["success"]:
                        status.update(label="Analysis Complete!", state="complete", expanded=False)
                    else:
                        status.update(label="Analysis Failed.", state="error", expanded=False)
            except Exception as e:
                elapsed_time = round(time.time() - start_time, 2)
                err_msg = str(e)
                err_msg_lower = err_msg.lower()
                exc_type = type(e).__name__
                model_sent = st.session_state.get("gemini_model", DEFAULT_MODEL)
                
                # Parse status code and model name if present in the error message
                status_code = None
                code_match = re.search(r"\b(\d{3})\b", err_msg)
                if code_match:
                    status_code = int(code_match.group(1))
                    
                gemini_error_code = None
                error_code_match = re.search(r"\(([A-Z_]+)\)", err_msg)
                if error_code_match:
                    gemini_error_code = error_code_match.group(1)
                    
                model_match = re.search(r"calling model '([^']+)'", err_msg)
                if model_match:
                    model_sent = model_match.group(1)

                # Exception Classification
                if status_code == 404 or "not_found" in err_msg_lower or "404" in err_msg:
                    st.error("Selected model is unavailable.")
                elif status_code == 403 or "permission_denied" in err_msg_lower or "403" in err_msg or "permission" in err_msg_lower:
                    st.error("Your API project does not have permission to use this model.")
                elif status_code == 401 or "api_key_invalid" in err_msg_lower or "invalid_api_key" in err_msg_lower or "invalid api key" in err_msg_lower or "401" in err_msg or "unauthorized" in err_msg_lower:
                    st.error("Invalid API key.")
                elif status_code == 429 or "resource_exhausted" in err_msg_lower or "quota" in err_msg_lower or "429" in err_msg:
                    st.error("API quota exceeded.")
                elif "connection" in err_msg_lower or "network" in err_msg_lower or "timeout" in err_msg_lower or "dns" in err_msg_lower or "socket" in err_msg_lower:
                    st.error("Unable to reach Gemini API.")
                else:
                    st.error(f"Unknown Error: {err_msg}")
                    
                if enable_debug:
                    st.markdown("### 🛠️ Debug Information")
                    st.write(f"**Model ID Sent**: `{model_sent}`")
                    
                    api_source = "None"
                    masked_key = "None"
                    if api_key:
                        env_key = os.getenv("GEMINI_API_KEY", "").strip()
                        if env_key and api_key == env_key:
                            api_source = "Environment Variable"
                        else:
                            api_source = "User Input"
                        
                        if len(api_key) > 8:
                            masked_key = api_key[:6] + "*" * (len(api_key) - 10) + api_key[-4:]
                        else:
                            masked_key = "*" * len(api_key)
                            
                    st.write(f"**API Source**: `{api_source}`")
                    st.write(f"**API Key (Masked)**: `{masked_key}`")
                    st.write(f"**HTTP Status**: `{status_code or 'Unknown'}`")
                    st.write(f"**Gemini Error Code**: `{gemini_error_code or 'Unknown'}`")
                    st.write(f"**Exception Type**: `{exc_type}`")
                    st.write(f"**Elapsed Time Before Failure**: `{elapsed_time}s`")
                    st.write("**First 500 Characters of Response**:")
                    st.code(err_msg[:500], language="text")
                    st.exception(e)
                st.stop()
                
            if not result["success"]:
                st.subheader("🤖 Analysis Findings")
                st.error("Analysis Failed")
                
                st.subheader("⚠️ Execution Error")
                if exec_result.get("stderr"):
                    st.code(exec_result["stderr"], language="text")
                else:
                    st.info("No execution error details available.")
                
                st.subheader("⚙️ Execution Details")
                col1, col2, col3 = st.columns(3)
                col1.metric("Execution Time", f"{exec_result.get('execution_time', 0)}s")
                col2.metric("Retries Used", result["retries"])
                col3.metric("Exit Status", exec_result.get("exit_status", "N/A"))
                
                st.subheader("🖼 Generated Visualizations")
                st.info("Visualization was not generated because code execution failed.")
            else:
                # Issue 8: Generated Code and Layout Order
                exec_summary = get_section(result["final_text"], "Executive Summary")
                key_insights = get_section(result["final_text"], "Key Insights")
                biz_recs = get_section(result["final_text"], "Business Recommendations")
                tech_summary = get_section(result["final_text"], "Technical Summary")
                
                clean_text_no_code = re.sub(r'```python\n.*?```', '', result["final_text"], flags=re.DOTALL).strip()
                extracted_content = "\n".join([exec_summary, key_insights, biz_recs, tech_summary]).strip()
                
                st.subheader("🤖 Analysis Findings")
                if len(extracted_content) < len(clean_text_no_code) * 0.4:
                    st.markdown(result["final_text"])
                else:
                    if exec_summary:
                        st.markdown(exec_summary)
                    if key_insights:
                        st.markdown(key_insights)
                    if biz_recs:
                        st.markdown(biz_recs)
                    if tech_summary:
                        st.markdown(tech_summary)
                
                st.subheader("💻 Executed Python Code")
                if result["current_code"]:
                    st.code(result["current_code"], language="python")
                else:
                    st.info("No Python code was generated.")
                
                st.subheader("⚙️ Execution Details")
                col1, col2, col3 = st.columns(3)
                col1.metric("Execution Time", f"{exec_result.get('execution_time', 0)}s")
                col2.metric("Retries Used", result["retries"])
                col3.metric("Exit Status", exec_result.get("exit_status", "N/A"))
                
                if exec_result.get("warnings"):
                    for w in exec_result["warnings"]:
                        st.warning(w)
                
                if exec_result["stdout"]:
                    with st.expander("📜 Standard Output", expanded=False):
                        st.code(exec_result["stdout"])
                    
                if exec_result["stderr"]:
                    with st.expander("⚠️ Standard Error", expanded=False):
                        st.code(exec_result["stderr"], language="text")
                    
                visualizations = [f for f in exec_result.get("files_generated", []) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.html'))]
                other_files = [f for f in exec_result.get("files_generated", []) if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.html'))]
                
                valid_visualizations = [f for f in visualizations if os.path.exists(f) and os.path.isfile(f)]
                
                if visualizations and not valid_visualizations:
                    st.subheader("🖼 Generated Visualizations")
                    st.info("Visualization was not generated because code execution failed.")
                elif valid_visualizations:
                    st.subheader("🖼 Generated Visualizations")
                    for file in valid_visualizations:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            st.image(file, caption=os.path.basename(file))
                        elif file.lower().endswith('.html'):
                            st.write(f"📊 Interactive Chart: {os.path.basename(file)}")
                            with open(file, 'r', encoding='utf-8') as f:
                                html_data = f.read()
                            st.components.v1.html(html_data, height=500, scrolling=True)
                
                valid_other_files = [f for f in other_files if os.path.exists(f) and os.path.isfile(f)]
                if valid_other_files:
                    st.subheader("📁 Generated Files")
                    for file in valid_other_files:
                        st.write(f"📁 Generated data file: {os.path.basename(file)}")
                        with open(file, "rb") as f:
                            st.download_button("Download Data", f, file_name=os.path.basename(file))
                
                # Auto Insights Panel (Feature 19) & Suggested Next Steps (Feature 15)
                st.markdown("---")
                st.subheader("💡 Business Insights Panel")
                st.info("The insights above indicate clear trends in the dataset. Use the download center to retrieve this context, or ask follow-up questions to explore further.")

                report = f"# Analysis Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                report += f"## User Request\n{user_question}\n\n"
                report += f"## Agent Response\n{result['final_text']}\n\n"
                if result["current_code"]:
                    report += f"## Code Executed\n```python\n{result['current_code']}\n```\n\n"
                report += f"## Execution Output\n```text\n{exec_result['stdout']}\n```\n"
                st.session_state.report_content += report
                
                st.download_button("📥 Export Report (Markdown)", st.session_state.report_content, file_name=f"{project_name}_report.md")
