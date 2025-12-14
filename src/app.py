import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# ============================
# Path setup (unchanged)
# ============================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "student_Ct_forecast_model.h5")
X_SCALER_PATH = os.path.join(ROOT_DIR, "models", "X_scaler.pkl")
Y_SCALER_PATH = os.path.join(ROOT_DIR, "models", "y_scaler.pkl")
STUDENT_CT_SCALER_PATH = os.path.join(ROOT_DIR, "models", "student_ct_scaler.pkl")
ENCODERS_PATH = os.path.join(ROOT_DIR, "models", "encoders.pkl")
BASE_CSV_PATH = os.path.join(ROOT_DIR, "forecasting_data", "Forecasting_streamlit_data.csv")
NIGERIA_GEOJSON = os.path.join(ROOT_DIR, "forecasting_data", "nigeria_lga.json")

# ============================
# Dashboard header
# ============================
st.set_page_config(page_title="Nigeria Education Forecast Dashboard", layout="wide")

st.title("🎓 Student Population Forecasting Across Nigerian LGAs")

# ↓ smaller font for overview section
st.markdown("""
<style>
.overview-text {font-size:15px; line-height:1.6; color:#ccc;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="overview-text">
This dashboard forecasts projected <b>student enrollment across Nigerian LGAs</b> using a deep learning model (CNN–LSTM).  
It leverages open geospatial, demographic, and educational indicators to estimate <b>future student population trends</b>.  
Use the controls in the sidebar to choose a forecast year or focus on specific states.
</div>
---
""", unsafe_allow_html=True)

# ----------------------------
# Load resources (cached)
# ----------------------------
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH, compile=False)
    with open(X_SCALER_PATH, "rb") as f:
        X_scaler = pickle.load(f)
    try:
        with open(STUDENT_CT_SCALER_PATH, "rb") as f:
            student_ct_scaler = pickle.load(f)
    except Exception:
        student_ct_scaler = None
    try:
        with open(Y_SCALER_PATH, "rb") as f:
            y_scaler = pickle.load(f)
    except Exception:
        y_scaler = None
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(NIGERIA_GEOJSON, "r") as f:
        nigeria_geojson = json.load(f)
    return model, X_scaler, student_ct_scaler, y_scaler, encoders, nigeria_geojson

model, X_scaler, student_ct_scaler, y_scaler, encoders, nigeria_geojson = load_resources()

# ----------------------------
# Sidebar controls (add info text)
# ----------------------------
st.sidebar.header("⚙️ Forecast Controls")
st.sidebar.markdown("Adjust parameters below to customize your forecast:")
year = st.sidebar.number_input("Select Forecast Year", min_value=2024, max_value=2035, value=2026, step=1)
selected_state = st.sidebar.text_input("Enter State Code (optional)", value="")
sample_n = st.sidebar.number_input("Max rows to forecast", min_value=0, max_value=100000, value=3000)
st.sidebar.caption("💡 Tip: Limit rows for faster computation on large datasets.")

# ----------------------------
# Load base data
# ----------------------------
@st.cache_data
def load_base_df():
    df = pd.read_csv(BASE_CSV_PATH)
    df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

base_df = load_base_df()
st.write(f"Base dataset loaded: {base_df.shape[0]:,} rows × {base_df.shape[1]} columns")

# ----------------------------
# Prepare df_forecast (same)
# ----------------------------
st.subheader("📊 Prepare data for forecasting")
selected_features = [
    'statecode', 'lganame', 'wardcode', 'FID', 'teacher_ct', 'source',
    'raster_value', 'education_Integrated', 'management_Private',
    'management_Public', 'management_State Government', 'management_Unknown',
    'subtype_Aggregate', 'subtype_Junior', 'subtype_Nursery',
    'subtype_Standard', 'subtype_Tertiary', 'category_Tertiary',
    'student_teacher_ratio', 'student_ct_lag1', 'AgeGroup', 'Year'
]
df_forecast = base_df.copy()
if selected_state:
    df_forecast = df_forecast[df_forecast["statecode"].astype(str) == str(selected_state)]
if "student_ct" in df_forecast.columns:
    df_forecast = df_forecast.drop(columns=["student_ct"])
df_forecast["Year"] = int(year)
if sample_n > 0 and len(df_forecast) > sample_n:
    st.info(f"Sampling {sample_n} rows from {len(df_forecast):,} for faster forecasting.")
    df_forecast = df_forecast.sample(sample_n, random_state=42).reset_index(drop=True)

# ----------------------------
# Safe categorical encoding
# ----------------------------
st.subheader("🔠 Encode categorical columns (safe)")
used_encoders = {}
for col, le in encoders.items():
    if col in df_forecast.columns:
        df_forecast[col] = df_forecast[col].astype(str)
        known = list(le.classes_)
        unseen = [x for x in df_forecast[col].unique() if x not in known and pd.notna(x)]
        if unseen:
            st.warning(f"Unseen labels in '{col}': {unseen[:5]}")
            le.classes_ = np.append(le.classes_, unseen)
        df_forecast[col] = df_forecast[col].apply(lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1)
        used_encoders[col] = le

# ----------------------------
# Align features
# ----------------------------
st.subheader("🧩 Align features to model/scaler expectation")
try:
    expected_features = list(X_scaler.feature_names_in_)
except Exception:
    expected_features = [c for c in selected_features if c in df_forecast.columns]
    st.warning("X_scaler missing feature names. Using fallback.")
df_forecast = df_forecast[[col for col in df_forecast.columns if col in expected_features]]
for col in expected_features:
    if col not in df_forecast.columns:
        df_forecast[col] = 0
df_forecast = df_forecast[expected_features]

# ----------------------------
# Run forecast
# ----------------------------
st.subheader("🔮 Run model and inverse-transform forecasts")
with st.spinner("Generating forecasts..."):
    X_scaled = X_scaler.transform(df_forecast)
    X_scaled_3d = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    forecast_scaled = model.predict(X_scaled_3d)
    forecast_scaled_2d = forecast_scaled.reshape(-1, 1)
    if student_ct_scaler is not None:
        forecast_original = student_ct_scaler.inverse_transform(forecast_scaled_2d).ravel()
    else:
        forecast_original = forecast_scaled_2d.ravel()

# ✅ Ensure non-negative whole-number forecasts
df_forecast_out = df_forecast.copy()
df_forecast_out["forecasted_student_population"] = np.maximum(forecast_original, 0)
df_forecast_out["forecasted_student_population"] = np.round(df_forecast_out["forecasted_student_population"]).astype(int)

# 🔢 Filter AgeGroup (≤ 19)
if "AgeGroup" in df_forecast_out.columns:
    df_forecast_out = df_forecast_out[df_forecast_out["AgeGroup"].astype(int) <= 19]

# ----------------------------
# Percentage growth from baseline (2024)
# ----------------------------
try:
    baseline = base_df[base_df["Year"] == 2024].groupby("lganame")["student_ct"].mean().reset_index()
    df_forecast_out = df_forecast_out.merge(baseline, on="lganame", how="left", suffixes=("", "_baseline"))
    df_forecast_out["pct_growth"] = (
        (df_forecast_out["forecasted_student_population"] - df_forecast_out["student_ct_baseline"])
        / df_forecast_out["student_ct_baseline"].replace(0, np.nan)
    ) * 100
    df_forecast_out["pct_growth"] = df_forecast_out["pct_growth"].fillna(0)
except Exception:
    df_forecast_out["pct_growth"] = 0

# ----------------------------
# Decode categorical columns
# ----------------------------
for col, le in used_encoders.items():
    if col in df_forecast_out.columns:
        df_forecast_out[col] = df_forecast_out[col].apply(
            lambda x: le.inverse_transform([x])[0] if 0 <= x < len(le.classes_) else "Unknown"
        )

# ----------------------------
# Results summary
# ----------------------------
st.success(f"✅ Forecasting complete — {len(df_forecast_out):,} records generated.")
avg_growth = df_forecast_out["pct_growth"].mean()
st.metric(label="📈 Average Forecast Growth (%)", value=f"{avg_growth:.2f}%")

# ----------------------------
# Interactive Map with growth toggle
# ----------------------------
st.subheader("🗺️ Interactive Forecast Map by LGA")
view_option = st.radio("Map view:", ["Forecasted Student Population", "Percentage Growth (%)"], horizontal=True)
map_color = "forecasted_student_population" if view_option == "Forecasted Student Population" else "pct_growth"
map_title = f"{view_option} per LGA ({year})"

if nigeria_geojson and "lganame" in df_forecast_out.columns:
    lga_summary = df_forecast_out.groupby("lganame", as_index=False)[map_color].mean()
    fig = px.choropleth_mapbox(
        lga_summary,
        geojson=nigeria_geojson,
        featureidkey="properties.NAME_2",
        locations="lganame",
        color=map_color,
        color_continuous_scale="Viridis",
        hover_name="lganame",
        hover_data={map_color: True},
        mapbox_style="carto-positron",
        zoom=4.6,
        center={"lat": 9.0820, "lon": 8.6753},
        opacity=0.7,
        title=map_title
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("GeoJSON or 'lganame' column missing.")

# ----------------------------
# Bar chart by LGA × AgeGroup
# ----------------------------
st.subheader("📊 Forecast by LGA and AgeGroup")
if ("lganame" in df_forecast_out.columns) and ("AgeGroup" in df_forecast_out.columns):
    lga_age_summary = df_forecast_out.groupby(["lganame", "AgeGroup"], as_index=False)["forecasted_student_population"].mean()
    fig2 = px.bar(
        lga_age_summary,
        x="lganame",
        y="forecasted_student_population",
        color="AgeGroup",
        title=f"Forecasted Student Population per LGA and AgeGroup ({year})",
        labels={"forecasted_student_population": "Forecasted Student Population"},
        height=600
    )
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("📄 View Forecast Table"):
        st.dataframe(lga_age_summary)

st.markdown("---")
st.caption("Developed by Feranmi Oyedare | Powered by CNN–LSTM Forecasting")
