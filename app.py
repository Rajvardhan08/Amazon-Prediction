import streamlit as st
import pandas as pd
import joblib
from haversine import haversine, Unit
import warnings

warnings.filterwarnings('ignore')

# --- Load Model and Columns ---
@st.cache_resource
def load_model_files():
    """Loads the trained model and the list of model columns from disk."""
    try:
        model = joblib.load('delivery_time_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except FileNotFoundError:
        return None, None

model, model_columns = load_model_files()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Delivery Time Predictor", layout="wide")
st.title('🚚 Delivery Time Prediction App')
st.markdown(
    "This application uses a machine learning model to predict delivery times. "
    "Use the sidebar to input the details of the delivery order."
)

# --- Sidebar for User Inputs ---
st.sidebar.header('Input Order Details')

user_inputs = {}

# --- Define General Input Fields ---
user_inputs['Agent_Age'] = st.sidebar.slider('Agent Age', 20, 50, 35)
user_inputs['Agent_Rating'] = st.sidebar.slider('Agent Rating', 1.0, 5.0, 4.5, 0.1)

user_inputs['Weather'] = st.sidebar.selectbox('Weather Condition', ['Sunny', 'Cloudy', 'Stormy', 'Sandstorms', 'Windy', 'Fog'])
user_inputs['Traffic'] = st.sidebar.selectbox('Traffic Condition', ['Low', 'Medium', 'High', 'Jam'])
user_inputs['Vehicle'] = st.sidebar.selectbox('Vehicle Type', ['motorcycle', 'scooter', 'electric_scooter', 'van'])
user_inputs['Area'] = st.sidebar.selectbox('Area Type', ['Urban', 'Metropolitian', 'Semi-Urban'])
user_inputs['Category'] = st.sidebar.selectbox('Order Category', ['Clothing', 'Electronics', 'Sports', 'Snacks', 'Drinks', 'Meal', 'Jewelry', 'Home', 'Skincare', 'Toys', 'Books', 'Grocery', 'Pet Supplies', 'Outdoors', 'Kitchen'])

# --- Conditional Input for Distance ---
st.sidebar.markdown("---")
input_method = st.sidebar.radio(
    "Choose how to input distance:",
    ('By Latitude/Longitude', 'Directly in Kilometers')
)

distance_km = 0

if input_method == 'By Latitude/Longitude':
    st.sidebar.subheader("Location Coordinates")
    # Use number_input for precise coordinate entry
    store_lat = st.sidebar.number_input('Store Latitude', value=22.74, format="%.4f")
    store_lon = st.sidebar.number_input('Store Longitude', value=75.89, format="%.4f")
    drop_lat = st.sidebar.number_input('Drop Latitude', value=22.76, format="%.4f")
    drop_lon = st.sidebar.number_input('Drop Longitude', value=75.91, format="%.4f")
    
    # Calculate distance for the model
    pickup_coords = (store_lat, store_lon)
    drop_coords = (drop_lat, drop_lon)
    distance_km = haversine(pickup_coords, drop_coords, unit=Unit.KILOMETERS)
    st.sidebar.info(f"Calculated Distance: **{distance_km:.2f} km**")

else:
    st.sidebar.subheader("Direct Distance")
    distance_km = st.sidebar.slider('Distance (km)', 1.0, 50.0, 10.0, 0.5)

user_inputs['Distance_km'] = distance_km


# --- Prediction Logic ---
if model is not None and model_columns is not None:
    if st.sidebar.button('Predict Delivery Time', use_container_width=True):
        
        input_df = pd.DataFrame([user_inputs])
        input_df_encoded = pd.get_dummies(input_df, dtype=int)
        final_input_df = input_df_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(final_input_df)[0]

        st.subheader('Prediction Result')
        st.metric(label="Predicted Delivery Time", value=f"{prediction:.2f} minutes")
else:
    st.error(
        "**Error: Model files not found.**\n\n"
        "Please make sure `delivery_time_model.pkl` and `model_columns.pkl` "
        "are in the same folder as this `app.py` file."
    )

