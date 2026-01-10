import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Paths and setup
# ----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "student_Ct_forecast_model.h5")
X_SCALER_PATH = os.path.join(ROOT_DIR, "models", "X_scaler.pkl")
Y_SCALER_PATH = os.path.join(ROOT_DIR, "models", "y_scaler.pkl")        
STUDENT_CT_SCALER_PATH = os.path.join(ROOT_DIR, "models", "student_ct_scaler.pkl")
ENCODERS_PATH = os.path.join(ROOT_DIR, "models", "encoders.pkl")
BASE_CSV_PATH = os.path.join(ROOT_DIR, "forecasting_data", "Forecasting_streamlit_data.csv")
NIGERIA_GEOJSON = os.path.join(ROOT_DIR, "forecasting_data", "nigeria_lga.json")

st.set_page_config(page_title="Nigeria Education Forecast Dashboard", layout="wide")
st.title("🎓 Nigeria Student Count & OOSC Forecast Dashboard")
st.markdown(
    "This dashboard shows the **forecasted number of students** and **out-of-school children (OOSC)** across Nigerian LGAs. "
    "It compares the forecast with a baseline estimate to show percentage increases or decreases."
)

# ----------------------------
# Load resources
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
# User controls
# ----------------------------
st.sidebar.header("⚙️ Forecast Options")
year = st.sidebar.number_input("Select Forecast Year", min_value=2024, max_value=2035, value=2026, step=1)
selected_state = st.sidebar.text_input("Filter by State Code (optional)", value="")
sample_n = st.sidebar.number_input(
    "Maximum number of rows to forecast (for speed)", min_value=0, max_value=100000, value=3000
)

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_base_df():
    df = pd.read_csv(BASE_CSV_PATH)
    df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

base_df = load_base_df()
st.write(f"Loaded dataset: {base_df.shape[0]:,} rows × {base_df.shape[1]} columns")

# ----------------------------
# Prepare forecast data
# ----------------------------
st.subheader("📊 Prepare data for forecasting")
st.markdown(
    "Filter and arrange the dataset, then compute a **baseline estimate** for student counts. "
    "Baseline = `attendance_prop_attending × WPP_value`"
)

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

# ----------------------------
# Restrict to school-age population (5–16 years)
# ----------------------------
if "AgeGroup" in df_forecast.columns:
    df_forecast["AgeGroup"] = pd.to_numeric(df_forecast["AgeGroup"], errors="coerce")
    df_forecast = df_forecast[
        (df_forecast["AgeGroup"] >= 5) &
        (df_forecast["AgeGroup"] <= 16)
    ]

# ----------------------------
# Compute baseline student count
# ----------------------------
st.subheader("📏 Baseline Student Count")

if 'attendance_prop_attending' in df_forecast.columns and 'WPP_value' in df_forecast.columns:
    WPP_scaled = df_forecast['WPP_value'].values.reshape(-1, 1)
    # Inverse scaling to get actual WPP values
    WPP_mean = 1000
    WPP_std = 200
    WPP_original = WPP_scaled * WPP_std + WPP_mean
    baseline_student_ct = (df_forecast['attendance_prop_attending'].values.reshape(-1, 1) * WPP_original).ravel()
else:
    st.warning("Missing columns for baseline calculation.")
    baseline_student_ct = np.nan * np.ones(len(df_forecast))
    WPP_original = np.nan * np.ones(len(df_forecast))

# Remove old student_ct if present
if "student_ct" in df_forecast.columns:
    df_forecast = df_forecast.drop(columns=["student_ct"])

df_forecast["Year"] = int(year)

# Sample for performance
if sample_n > 0 and len(df_forecast) > sample_n:
    st.info(f"Sampling {sample_n} rows for faster processing.")
    df_forecast = df_forecast.sample(sample_n, random_state=42).reset_index(drop=True)
    baseline_student_ct = baseline_student_ct[df_forecast.index]
    WPP_original = WPP_original[df_forecast.index]

# ----------------------------
# Encode categorical columns safely
# ----------------------------
st.subheader("🔠 Encode categories for model")
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
# Align features for model
# ----------------------------
st.subheader("🧩 Align features for model")
try:
    expected_features = list(X_scaler.feature_names_in_)
except Exception:
    expected_features = [c for c in selected_features if c in df_forecast.columns]
    st.warning("X_scaler has no feature names; using fallback.")

df_forecast = df_forecast[[col for col in df_forecast.columns if col in expected_features]]
for col in expected_features:
    if col not in df_forecast.columns:
        df_forecast[col] = 0
df_forecast = df_forecast[expected_features]

# ----------------------------
# Run forecast
# ----------------------------
st.subheader("🔮 Generate forecast")
with st.spinner("Generating forecasts..."):
    X_scaled = X_scaler.transform(df_forecast)
    X_scaled_3d = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    forecast_scaled = model.predict(X_scaled_3d)
    forecast_scaled_2d = forecast_scaled.reshape(-1, 1)
    if student_ct_scaler is not None:
        forecast_original = student_ct_scaler.inverse_transform(forecast_scaled_2d).ravel()
    else:
        forecast_original = forecast_scaled_2d.ravel()

# ----------------------------
# Compute OOSC and % changes
# ----------------------------
df_forecast_out = df_forecast.copy()
df_forecast_out["Predicted Student Count"] = np.round(forecast_original).astype(int)
df_forecast_out["Baseline Student Count"] = np.round(baseline_student_ct).astype(int)
df_forecast_out["OOSC_Predicted"] = np.round(WPP_original.ravel() - df_forecast_out["Predicted Student Count"]).astype(int)
df_forecast_out["OOSC_Baseline"] = np.round(WPP_original.ravel() - df_forecast_out["Baseline Student Count"]).astype(int)

# Percentage Change in Student Count
df_forecast_out["Percentage Change in Student Count"] = np.where(
    df_forecast_out["Baseline Student Count"] == 0,
    0,
    ((df_forecast_out["Predicted Student Count"] - df_forecast_out["Baseline Student Count"])
     / df_forecast_out["Baseline Student Count"] * 100)
).round(2)

# Percentage Change in OOSC
df_forecast_out["Percentage Change in OOSC"] = np.where(
    df_forecast_out["OOSC_Baseline"] == 0,
    0,
    ((df_forecast_out["OOSC_Predicted"] - df_forecast_out["OOSC_Baseline"])
     / df_forecast_out["OOSC_Baseline"] * 100)
).round(2)

# ----------------------------
# Decode categories for display
# ----------------------------
for col, le in used_encoders.items():
    if col in df_forecast_out.columns:
        df_forecast_out[col] = df_forecast_out[col].apply(
            lambda v: le.inverse_transform([v])[0] if 0 <= v < len(le.classes_) else "Unknown"
        )

if "AgeGroup" in df_forecast_out.columns:
    df_forecast_out["AgeGroup"] = df_forecast_out["AgeGroup"].astype(str)

st.success(f"✅ Forecast complete — {len(df_forecast_out)} rows.")

# ----------------------------
# Display sample table
# ----------------------------
st.subheader("📈 Forecast Summary (sample)")
show_cols = [
    "statecode", "lganame", "wardcode", "AgeGroup", "Year",
    "Predicted Student Count", "Baseline Student Count",
    "OOSC_Predicted", "OOSC_Baseline",
    "Percentage Change in Student Count", "Percentage Change in OOSC"
]
st.dataframe(df_forecast_out[show_cols].head(20))

# ----------------------------
# Map visualization for Percentage Change in OOSC
# ----------------------------
st.subheader("🗺️ Map of Percentage Change in OOSC")
st.markdown("Green = decrease in OOSC, Red = increase in OOSC compared to baseline")
if nigeria_geojson is None:
    st.warning("GeoJSON not loaded — can't render map.")
else:
    if "lganame" in df_forecast_out.columns:
        lga_summary = df_forecast_out.groupby(["statecode", "lganame"], as_index=False)["Percentage Change in OOSC"].mean()
        try:
            fig = px.choropleth_mapbox(
                lga_summary,
                geojson=nigeria_geojson,
                featureidkey="properties.NAME_2",
                locations="lganame",
                color="Percentage Change in OOSC",
                hover_name="lganame",
                hover_data={"statecode": True, "Percentage Change in OOSC": ":.2f"},
                color_continuous_scale="RdYlGn_r",  # red = increase, green = decrease
                mapbox_style="carto-positron",
                zoom=4.6,
                center={"lat": 9.0820, "lon": 8.6753},
                opacity=0.7,
                title=f"Percentage Change in OOSC per LGA ({year})"
            )
            fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, hoverlabel=dict(bgcolor="white", font_size=13))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Map rendering error: {e}")

# ----------------------------
# Aggregated bar chart: Percentage Change in OOSC by LGA and AgeGroup
# ----------------------------
st.subheader("📊 Percentage Change in OOSC by LGA and AgeGroup")
if ("lganame" in df_forecast_out.columns) and ("AgeGroup" in df_forecast_out.columns):
    lga_age_summary = df_forecast_out.groupby(["lganame", "AgeGroup"], as_index=False)["Percentage Change in OOSC"].mean().round(2)
    fig2 = px.bar(
        lga_age_summary,
        x="lganame",
        y="Percentage Change in OOSC",
        color="AgeGroup",
        title=f"Percentage Change in OOSC per LGA and AgeGroup ({year})",
        labels={"Percentage Change in OOSC": "% Change in OOSC"},
        height=600
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Insufficient data to show LGA × AgeGroup chart.")

# ----------------------------
# Full table expander
# ----------------------------
with st.expander("📄 View full table"):
    st.dataframe(df_forecast_out[show_cols])

st.markdown("---")
st.caption("Developed by Feranmi Oyedare | Powered by CNN–LSTM Forecasting")