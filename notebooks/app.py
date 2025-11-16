import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------
# REQUIRED: Custom Object Definitions
# (We must define these so joblib.load() can understand the pipeline)
# --------------------------------------------------------------------------

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features as their frequency."""
    def __init__(self):
        self.freq_map_ = {}
        self.default_ = 0.0

    def fit(self, X, y=None):
        s = pd.Series(X.ravel()) if hasattr(X, "ravel") else pd.Series(X)
        self.freq_map_ = (s.value_counts(normalize=True)).to_dict()
        self.default_ = 0.0  # Default for unseen values
        return self

    def transform(self, X):
        s = pd.Series(X.ravel()) if hasattr(X, "ravel") else pd.Series(X)
        return s.map(self.freq_map_).fillna(self.default_).to_numpy().reshape(-1, 1)

def convert_bool_to_int(X):
    """Converts boolean array to integer array for the pipeline."""
    return X.astype(int)

# --------------------------------------------------------------------------
# Load The Model Pipeline
# --------------------------------------------------------------------------

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Loads the pickled pipeline object."""
    try:
        pipeline = joblib.load("full_pipeline.pkl")
        return pipeline
    except FileNotFoundError:
        st.error("Model file 'full_pipeline.pkl' not found.")
        st.info("Please make sure the file is in the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the pipeline
pipeline = load_model()

# --------------------------------------------------------------------------
# Page Configuration & UI
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Electricity Production Predictor",
    page_icon="💡",
    layout="wide"
)

# --- Sidebar for Inputs ---
st.sidebar.header("Input Features")
st.sidebar.write("Adjust the values to predict the production level.")

# Create input widgets for all 14 features
# We base the options on your training script examples

# num_cols
year = st.sidebar.number_input("Year", 2000, 2030, 2024, 1)
month = st.sidebar.slider("Month", 1, 12, 11)
quarter = st.sidebar.slider("Quarter", 1, 4, 4)
st.sidebar.caption("Note: 'Month' and 'Quarter' are separate inputs for the model.")

# binary_cols
is_renewable = st.sidebar.selectbox("Is Renewable Source?", [True, False], index=0)
is_fossil = st.sidebar.selectbox("Is Fossil Fuel Source?", [True, False], index=1)
is_developed = st.sidebar.selectbox("Is Developed Country?", [True, False], index=0)
is_zero = st.sidebar.selectbox("Is Zero Emission?", [True, False], index=1)

# freq_cols
country_clean = st.sidebar.text_input("Country", "Sri Lanka")

# onehot_cols
season = st.sidebar.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Autumn'])
energy_source_clean = st.sidebar.text_input("Energy Source", "Hydro")
energy_type = st.sidebar.selectbox("Energy Type", ['Renewable', 'Conventional', 'Fossil Fuel'])
source_category = st.sidebar.selectbox("Source Category", ['Primary', 'Fossil Fuel'])
year_category = st.sidebar.selectbox("Year Category", ['Late 2010s', 'Before 2010', 'Early 2020s'])

# ordinal_cols
energy_intensity_category = st.sidebar.selectbox(
    "Energy Intensity", 
    ['Low', 'Medium', 'High', 'Very High'] # Must match the order from your pipeline
)

# --- Main Page for Outputs ---
st.title("💡 Electricity Production Predictor")
st.write("This app predicts whether monthly electricity production will be **High** or **Low** based on global energy source trends. Use the sidebar to enter your data.")

# Create the "Predict" button
if st.sidebar.button("Predict"):
    if pipeline:
        # 1. Create a DataFrame from the inputs
        input_data = {
            'year': year,
            'month': month,
            'quarter': quarter,
            'is_renewable': is_renewable,
            'is_fossil': is_fossil,
            'is_developed': is_developed,
            'is_zero': is_zero,
            'country_clean': country_clean,
            'season': season,
            'energy_source_clean': energy_source_clean,
            'energy_type': energy_type,
            'source_category': source_category,
            'year_category': year_category,
            'energy_intensity_category': energy_intensity_category
        }
        input_df = pd.DataFrame([input_data])
        
        st.subheader("Prediction Result")
        
        try:
            # 2. Get prediction
            pred_class = pipeline.predict(input_df)[0]
            
            # 3. Get probabilities
            # proba[0] is for class 0 (Low), proba[1] is for class 1 (High)
            pred_proba = pipeline.predict_proba(input_df)[0]
            
            # 4. Display the result
            if pred_class == 1:
                st.success("Prediction: HIGH Production")
            else:
                st.error("Prediction: LOW Production")
            
            # 5. Display probabilities in columns
            col1, col2 = st.columns(2)
            col1.metric(
                label="Probability of HIGH Production",
                value=f"{pred_proba[1] * 100:.2f}%"
            )
            col2.metric(
                label="Probability of LOW Production",
                value=f"{pred_proba[0] * 100:.2f}%"
            )
            
            # 6. Show the input data used for the prediction
            st.write("---")
            st.subheader("Inputs Used for This Prediction:")
            # Transpose the DataFrame for better readability
            st.dataframe(input_df.T.rename(columns={0: 'Input Value'}))
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please check your inputs. For text fields like 'Country', ensure they match what the model expects.")
    else:
        st.warning("Model is not loaded. Cannot make predictions.")