# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# ============================
# Config - update these paths to match your environment
# ============================
MODEL_PATH = r"/Users/user/Desktop/AIPISCIAA/notebooks/student_Ct_forecast_model.h5"
X_SCALER_PATH = r"/Users/user/Desktop/AIPISCIAA/notebooks/models/X_scaler.pkl"
Y_SCALER_PATH = r"/Users/user/Desktop/AIPISCIAA/notebooks/models/y_scaler.pkl"        # optional
STUDENT_CT_SCALER_PATH = r"/Users/user/Desktop/AIPISCIAA/notebooks/student_ct_scaler.pkl"
ENCODERS_PATH = r"/Users/user/Desktop/AIPISCIAA/notebooks/models/encoders.pkl"
BASE_CSV_PATH = r"/Users/user/Desktop/AIPISCIAA/data/scenario_2_dataset.csv"
NIGERIA_GEOJSON = r"/Users/user/Desktop/AIPISCIAA/src/nigeria_lga.json"
# ============================

st.set_page_config(page_title="Nigeria Education Forecast Dashboard", layout="wide")
st.title("🎓 Student Count Forecasting Across Nigerian LGAs")

# ----------------------------
# Load resources (cached)
# ----------------------------
@st.cache_resource
def load_resources():
    # Model
    try:
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Could not load model at {MODEL_PATH}: {e}")
        raise

    # Scalers & encoders
    try:
        with open(X_SCALER_PATH, "rb") as f:
            X_scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Could not load X_scaler at {X_SCALER_PATH}: {e}")
        raise

    # y_scaler is optional / legacy: we prefer student_ct_scaler
    y_scaler = None
    try:
        with open(Y_SCALER_PATH, "rb") as f:
            y_scaler = pickle.load(f)
    except Exception:
        y_scaler = None

    student_ct_scaler = None
    try:
        with open(STUDENT_CT_SCALER_PATH, "rb") as f:
            student_ct_scaler = pickle.load(f)
    except Exception:
        student_ct_scaler = None

    encoders = {}
    try:
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
            # sanity: ensure encoders is a dict of LabelEncoders
            if not isinstance(encoders, dict):
                raise ValueError("encoders.pkl does not contain a dict")
    except Exception as e:
        st.warning(f"Could not load encoders.pkl: {e}. Continuing without encoders (categoricals will remain as strings).")
        encoders = {}

    # geojson
    nigeria_geojson = None
    try:
        with open(NIGERIA_GEOJSON, "r") as f:
            nigeria_geojson = json.load(f)
    except Exception:
        nigeria_geojson = None

    return model, X_scaler, student_ct_scaler, y_scaler, encoders, nigeria_geojson

model, X_scaler, student_ct_scaler, y_scaler, encoders, nigeria_geojson = load_resources()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("⚙️ Forecast Controls")
year = st.sidebar.number_input("Select Forecast Year", min_value=2024, max_value=2035, value=2026, step=1)
selected_state = st.sidebar.text_input("Enter State Code (optional)", value="")  # empty => all states
sample_n = st.sidebar.number_input("Max rows to forecast (0 = all)", min_value=0, max_value=5000, value=3000)

# ----------------------------
# Load base data (cached)
# ----------------------------
@st.cache_data
def load_base_df():
    df = pd.read_csv(BASE_CSV_PATH)
    # drop duplicate-suffixed column names like 'wardname.13' if they exist, by removing trailing ".\d+"
    df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
    # drop duplicate columns that remain
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

# optional state filter
if selected_state:
    df_forecast = df_forecast[df_forecast["statecode"].astype(str) == str(selected_state)]

# drop target if present
if "student_ct" in df_forecast.columns:
    df_forecast = df_forecast.drop(columns=["student_ct"])

# set forecast year
df_forecast["Year"] = int(year)

# sample for performance
if sample_n > 0 and len(df_forecast) > sample_n:
    st.info(f"Sampling {sample_n} rows from {len(df_forecast):,} for faster forecasting.")
    df_forecast = df_forecast.sample(sample_n, random_state=42).reset_index(drop=True)

st.write("Data ready for forecasting:", df_forecast.shape)

# ----------------------------
# Safe categorical encoding using saved encoders
# ----------------------------
st.subheader("🔠 Encode categorical columns (safe)")

# We'll collect encoders used so we can decode later
used_encoders = {}

for col, le in encoders.items():
    if col in df_forecast.columns:
        # ensure string
        df_forecast[col] = df_forecast[col].astype(str)

        # identify unseen labels
        known = list(le.classes_)
        unseen = [x for x in df_forecast[col].unique() if x not in known and pd.notna(x)]
        if unseen:
            st.warning(f"Unseen labels found in '{col}': {unseen[:5]} (added to encoder for transform).")
            # extend encoder classes_ to include unseen labels so transform won't KeyError
            le.classes_ = np.append(le.classes_, unseen)

        # transform: if any item is NaN or '', map to -1
        def safe_transform(val):
            if pd.isna(val):
                return -1
            try:
                return int(le.transform([val])[0])
            except Exception:
                # if still fails, append and transform
                le.classes_ = np.append(le.classes_, [val])
                return int(le.transform([val])[0])

        df_forecast[col] = df_forecast[col].apply(safe_transform)
        used_encoders[col] = le

# ----------------------------
# Align features to what the X_scaler expects
# ----------------------------
st.subheader("🧩 Align features to model/scaler expectation")

# Determine expected feature names
try:
    expected_features = list(X_scaler.feature_names_in_)
except Exception:
    # fallback: if scaler was fit on a DataFrame and doesn't store names, try to use selected_features intersection
    expected_features = [c for c in selected_features if c in df_forecast.columns]
    st.warning("X_scaler has no feature_names_in_. Falling back to selected_features intersection.")

st.write(f"Scaler expects {len(expected_features)} features.")

# Drop columns not seen during fit
df_forecast = df_forecast[[col for col in df_forecast.columns if col in expected_features]]

# Add missing features as zeros (numeric)
for col in expected_features:
    if col not in df_forecast.columns:
        df_forecast[col] = 0

# Reorder to match expected_features
df_forecast = df_forecast[expected_features]

st.write("Aligned data shape:", df_forecast.shape)

# ----------------------------
# Scale, predict and inverse-transform forecast
# ----------------------------
st.subheader("🔮 Run model and inverse-transform forecasts")

with st.spinner("Generating forecasts..."):
    # scale X
    X_scaled = X_scaler.transform(df_forecast)  # returns 2D array
    # reshape for Conv1D+LSTM: [samples, timesteps, features_per_timestep]
    X_scaled_3d = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # predict (scaled / model output)
    forecast_scaled = model.predict(X_scaled_3d)

    # ensure 2D for inverse_transform compatibility
    forecast_scaled_2d = forecast_scaled.reshape(-1, 1)

    # try to inverse-transform to the original student_ct scale
    forecast_original = None
    if student_ct_scaler is not None:
        try:
            forecast_original = student_ct_scaler.inverse_transform(forecast_scaled_2d).ravel()
            st.success("✅ Forecasts converted back to original student_ct scale using student_ct_scaler.pkl")
        except Exception as e:
            st.warning(f"Could not inverse-transform with student_ct_scaler: {e}")
            forecast_original = forecast_scaled_2d.ravel()
    elif y_scaler is not None:
        try:
            forecast_original = y_scaler.inverse_transform(forecast_scaled_2d).ravel()
            st.info("Used y_scaler as fallback to inverse-transform forecasts.")
        except Exception as e:
            st.warning(f"Could not inverse-transform with y_scaler: {e}")
            forecast_original = forecast_scaled_2d.ravel()
    else:
        st.warning("No student_ct scaler or y_scaler available; returning scaled predictions.")
        forecast_original = forecast_scaled_2d.ravel()

# Attach forecast to df
df_forecast_out = df_forecast.copy()
df_forecast_out["forecast_student_ct"] = forecast_original

# ----------------------------
# Decode categorical columns back to names (safe)
# ----------------------------
st.subheader("🔁 Decode categorical columns (for display)")

for col, le in used_encoders.items():
    if col in df_forecast_out.columns:
        # df currently contains encoded ints; decode where valid (>=0)
        # values that are -1 (or out of range) will be set to "Unknown"
        decoded = []
        n_classes = len(le.classes_)
        for v in df_forecast_out[col].astype(int).values:
            if v >= 0 and v < n_classes:
                try:
                    decoded.append(le.inverse_transform([v])[0])
                except Exception:
                    decoded.append("Unknown")
            else:
                decoded.append("Unknown")
        df_forecast_out[col] = decoded

# Ensure AgeGroup is readable (if encoded earlier as numbers)
if "AgeGroup" in df_forecast_out.columns:
    # AgeGroup may now be strings (decoded) or ints; keep as string for plotting/coloring
    df_forecast_out["AgeGroup"] = df_forecast_out["AgeGroup"].astype(str)

st.success(f"✅ Forecasting complete — produced {len(df_forecast_out)} forecast rows.")

# ----------------------------
# Display results: sample table
# ----------------------------
st.subheader("📈 Forecast Summary (sample)")
show_cols = [c for c in ["statecode", "lganame", "wardcode", "AgeGroup", "Year", "forecast_student_ct"] if c in df_forecast_out.columns]
if len(df_forecast_out) == 0:
    st.warning("No rows to display after filtering/sampling.")
else:
    df_display = df_forecast_out[show_cols].copy()
    # Round forecast column for readability
    df_display["forecast_student_ct"] = df_display["forecast_student_ct"].round(2)
    st.dataframe(df_display.head(20))

# ----------------------------
# Map visualization (choropleth) if geojson present
# ----------------------------
st.subheader("🗺️ Forecast Map by LGA")
if nigeria_geojson is None:
    st.warning("Nigeria geojson not loaded — can't render map.")
else:
    # aggregate to LGA level average forecast
    if "lganame" in df_forecast_out.columns:
        lga_summary = df_forecast_out.groupby("lganame", as_index=False)["forecast_student_ct"].mean()
        # choropleth_mapbox expects the location field to match a feature property in the geojson
        try:
            fig = px.choropleth_mapbox(
                lga_summary,
                geojson=nigeria_geojson,
                featureidkey="properties.LGAName",
                locations="lganame",
                color="forecast_student_ct",
                color_continuous_scale="Viridis",
                mapbox_style="carto-positron",
                zoom=4.5,
                center={"lat": 9.0820, "lon": 8.6753},
                opacity=0.7,
                title=f"Forecasted Student Count per LGA ({year})"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not produce choropleth: {e}")
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
