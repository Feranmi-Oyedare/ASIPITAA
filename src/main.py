import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


ABS_PATH = os.path.abspath(r"/Users/user/Desktop/AIPISCIAA/")

# ============================
# Config - update these paths to match your environment
# ============================
MODEL_PATH = ABS_PATH + "/models/student_Ct_forecast_model.h5"
X_SCALER_PATH = ABS_PATH + "/models/X_scaler.pkl"
Y_SCALER_PATH = ABS_PATH + "/models/y_scaler.pkl"        
STUDENT_CT_SCALER_PATH = ABS_PATH + "/models/student_ct_scaler.pkl"
ENCODERS_PATH = ABS_PATH + "/models/encoders.pkl"
BASE_CSV_PATH = ABS_PATH + "/forecasting_data/Forecasting_streamlit_data.csv"
NIGERIA_GEOJSON = ABS_PATH + "/forecasting_data/nigeria_lga.json"
# ============================

st.set_page_config(page_title="Nigeria Education Forecast Dashboard", layout="wide")
st.title("🎓 Student Count Forecasting Across Nigerian LGAs")

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
# Sidebar controls
# ----------------------------
st.sidebar.header("⚙️ Forecast Controls")
year = st.sidebar.number_input("Select Forecast Year", min_value=2024, max_value=2035, value=2026, step=1)
selected_state = st.sidebar.text_input("Enter State Code (optional)", value="")
sample_n = st.sidebar.number_input("Max rows to forecast", min_value=0, max_value=100000, value=3000)

# ----------------------------
# Load base data (cached)
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
# Prepare df_forecast
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

# st.write("Data ready for forecasting:", df_forecast.shape)

# ----------------------------
# Safe categorical encoding
# ----------------------------
st.subheader("🔠 Encode categorical columns (safe)", )

used_encoders = {}
for col, le in encoders.items():
    if col in df_forecast.columns:
        df_forecast[col] = df_forecast[col].astype(str)
        known = list(le.classes_)
        unseen = [x for x in df_forecast[col].unique() if x not in known and pd.notna(x)]
        if unseen:
            st.warning(f"Unseen labels in '{col}': {unseen[:5]}")
            le.classes_ = np.append(le.classes_, unseen)
        def safe_transform(val):
            if pd.isna(val):
                return -1
            try:
                return int(le.transform([val])[0])
            except Exception:
                le.classes_ = np.append(le.classes_, [val])
                return int(le.transform([val])[0])
        df_forecast[col] = df_forecast[col].apply(safe_transform)
        used_encoders[col] = le

# ----------------------------
# Align features
# ----------------------------
st.subheader("🧩 Align features to model/scaler expectation")

try:
    expected_features = list(X_scaler.feature_names_in_)
except Exception:
    expected_features = [c for c in selected_features if c in df_forecast.columns]
    st.warning("X_scaler has no feature_names_in_. Using fallback.")

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
        st.success("✅ Forecasts restored to original student_ct scale.")
    else:
        forecast_original = forecast_scaled_2d.ravel()

df_forecast_out = df_forecast.copy()
df_forecast_out["forecast_student_ct"] = forecast_original

# ----------------------------
# Decode categorical columns
# ----------------------------
st.subheader("🔁 Decode categorical columns (for display)")
for col, le in used_encoders.items():
    if col in df_forecast_out.columns:
        decoded = []
        for v in df_forecast_out[col].astype(int).values:
            if 0 <= v < len(le.classes_):
                decoded.append(le.inverse_transform([v])[0])
            else:
                decoded.append("Unknown")
        df_forecast_out[col] = decoded

if "AgeGroup" in df_forecast_out.columns:
    df_forecast_out["AgeGroup"] = df_forecast_out["AgeGroup"].astype(str)

st.success(f"✅ Forecasting complete — {len(df_forecast_out)} rows.")

# ----------------------------
# Display sample table
# ----------------------------
st.subheader("📈 Forecast Summary (sample)")
show_cols = [c for c in ["statecode", "lganame", "wardcode", "AgeGroup", "Year", "forecast_student_ct"] if c in df_forecast_out.columns]
df_display = df_forecast_out[show_cols].copy()
df_display["forecast_student_ct"] = df_display["forecast_student_ct"].round(2)
st.dataframe(df_display.head(20))

# ----------------------------
# 🗺️ INTERACTIVE MAP (Enhanced)
# ----------------------------
st.subheader("🗺️ Interactive Forecast Map by LGA")
if nigeria_geojson is None:
    st.warning("GeoJSON not loaded — can't render map.")
else:
    if "lganame" in df_forecast_out.columns:
        lga_summary = df_forecast_out.groupby(["statecode", "lganame"], as_index=False)["forecast_student_ct"].mean()
        try:
            # Enhanced interactivity: hover info, tooltips, zoom, and color clarity
            fig = px.choropleth_mapbox(
                lga_summary,
                geojson=nigeria_geojson,
                featureidkey="properties.NAME_2",  # changed for correct LGA name match
                locations="lganame",
                color="forecast_student_ct",
                hover_name="lganame",
                hover_data={"statecode": True, "forecast_student_ct": ":.0f"},
                color_continuous_scale="Viridis",
                mapbox_style="carto-positron",
                zoom=4.6,
                center={"lat": 9.0820, "lon": 8.6753},
                opacity=0.7,
                title=f"Forecasted Student Count per LGA ({year})"
            )
            fig.update_layout(
                margin={"r":0,"t":40,"l":0,"b":0},
                mapbox_accesstoken=None,
                hoverlabel=dict(bgcolor="white", font_size=13),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Map rendering error: {e}")
    else:
        st.warning("Column 'lganame' not available for mapping.")

# ----------------------------
# Bar chart by LGA × AgeGroup
# ----------------------------
st.subheader("📊 Forecast by LGA and AgeGroup")
if ("lganame" in df_forecast_out.columns) and ("AgeGroup" in df_forecast_out.columns):
    lga_age_summary = df_forecast_out.groupby(["lganame", "AgeGroup"], as_index=False)["forecast_student_ct"].mean()
    fig2 = px.bar(
        lga_age_summary,
        x="lganame",
        y="forecast_student_ct",
        color="AgeGroup",
        title=f"Forecasted Student Count per LGA and AgeGroup ({year})",
        labels={"forecast_student_ct": "Forecasted Student Count"},
        height=600
    )
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("📄 View Forecast Table"):
        st.dataframe(lga_age_summary)
else:
    st.info("Insufficient columns to show LGA × AgeGroup chart.")

st.markdown("---")
st.caption("Developed by Feranmi Oyedare | Powered by CNN–LSTM Forecasting")
