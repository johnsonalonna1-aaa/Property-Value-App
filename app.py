import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# App settings
# ---------------------------
st.set_page_config(page_title="Property Value Predictor", layout="wide")
st.title("Hamilton County Property Value Predictor")
st.write("Predict **APPRAISED_VALUE** using a regression model trained on the dataset.")
st.caption("Disclaimer: Educational use only.")

DATA_FILE = "Housing_small.xlsx"
TARGET = "APPRAISED_VALUE"

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]  # clean column names
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure target is numeric and valid
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]

    return df

def get_numeric_X_y(df: pd.DataFrame):
    # Keep only numeric columns (simple + reliable)
    num = df.select_dtypes(include=["number"]).copy()

    if TARGET not in num.columns:
        st.error(f"'{TARGET}' is not found as a numeric column after cleaning.")
        st.stop()

    # Drop common ID-like columns if present
    id_words = ["ID", "PARCEL", "PIN", "ACCOUNT", "OBJECTID"]
    drop_cols = [c for c in num.columns if any(w in c.upper() for w in id_words)]
    num = num.drop(columns=drop_cols, errors="ignore")

    X = num.drop(columns=[TARGET])
    y = num[TARGET]

    if X.shape[1] == 0:
        st.error("No numeric feature columns found to train the model.")
        st.stop()

    return X, y

# ---------------------------
# Main
# ---------------------------
try:
    raw = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(f"Could not find **{DATA_FILE}** in the same folder as app.py.")
    st.stop()

st.subheader("1) Dataset Preview")
st.write("Shape:", raw.shape)
st.dataframe(raw.head())

df = clean_data(raw)

st.subheader("2) After Cleaning")
st.write("Shape:", df.shape)

X, y = get_numeric_X_y(df)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

st.subheader("3) Model Performance")
c1, c2 = st.columns(2)
c1.metric("MAE (lower is better)", f"{mae:,.0f}")
c2.metric("R² (closer to 1 is better)", f"{r2:.3f}")

st.write("Features used:")
st.write(list(X.columns))

# ---------------------------
# User Inputs + Prediction
# ---------------------------
st.subheader("4) Enter Inputs → Predict APPRAISED_VALUE")
st.sidebar.header("Inputs")

user_input = {}

for col in X.columns:
    col_series = X_train[col].dropna()

    # If a column is constant or empty, skip it
    if col_series.empty or col_series.nunique() == 1:
        continue

    col_min = float(col_series.min())
    col_max = float(col_series.max())
    col_mean = float(col_series.mean())

    # Use int slider if column looks like integers
    if (col_series % 1 == 0).all():
        user_input[col] = st.sidebar.slider(
            col,
            int(col_min),
            int(col_max),
            int(col_mean),
            step=1
        )
    else:
        user_input[col] = st.sidebar.slider(
            col,
            col_min,
            col_max,
            col_mean
        )

input_df = pd.DataFrame([user_input])

st.write("Your inputs:")
st.dataframe(input_df)

prediction = model.predict(input_df)[0]

st.markdown("---")
st.subheader("Predicted APPRAISED_VALUE")
st.write(f"**${prediction:,.0f}**")
