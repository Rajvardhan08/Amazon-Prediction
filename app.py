import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Amazon Delivery Time Predictor",
    page_icon="🚚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Model and Columns ---
# Use a caching decorator to load the model only once
@st.cache_resource
def load_model_files():
    """Loads the pre-trained model and required columns from disk."""
    # Check if model files exist
    if not os.path.exists('delivery_time_model.pkl') or not os.path.exists('model_columns.pkl'):
        st.error("Model files not found. Please ensure `delivery_time_model.pkl` and `model_columns.pkl` are in the same directory as this app.")
        st.stop()
    
    # Load the files
    model = joblib.load('delivery_time_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

model, model_columns = load_model_files()

# --- Application Header ---
st.title("🚚 Amazon Delivery Time Prediction")
st.markdown("""
This application predicts the delivery time for an order based on various factors. 
Use the sidebar on the left to input the details of a delivery, and the model will estimate the time required in minutes.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Delivery Details")

def get_user_inputs():
    """Creates sidebar widgets and returns a DataFrame of user inputs."""
    
    age = st.sidebar.slider('Delivery Person Age', 18, 50, 25)
    ratings = st.sidebar.slider('Delivery Person Rating', 1.0, 5.0, 4.5, 0.1)
    distance = st.sidebar.slider('Distance (km)', 0.0, 50.0, 5.0, 0.1)
    store_longitude = st.sidebar.slider('Store Longitude', 0.0, 180.0, 90.0, 0.5)
    store_latitude = st.sidebar.slider('Store Latitude', 0.0, 180.0, 90.0, 0.5)
    drop_longitude = st.sidebar.slider('Drop Longitude', 0.0, 180.0, 90.0, 0.5)
    drop_latitude = st.sidebar.slider('Drop Latitude', 0.0, 180.0, 90.0, 0.5)

    
    weather = st.sidebar.selectbox('Weather Conditions', 
        ['Sunny', 'Stormy', 'Sandstorms', 'Windy', 'Fog', 'Cloudy'])
        
    traffic = st.sidebar.selectbox('Road Traffic Density', 
        ['Low', 'Medium', 'Jam', 'High'])
        
    # vehicle_condition = st.sidebar.selectbox('Vehicle Condition', 
    #     ['Excellent', 'Good', 'Poor'])
        
    order_type = st.sidebar.selectbox('Type of Order', 
        ['Clothing', 'Electronics', 'Sports', 'Cosmetics', 'Toys', 'Apparel', 'Books', 'Home', 'Grocery', 'Jewelry', 'Kitchen', 'Outdoors', 'Pet Supplies', 'Shoes', 'Skincare', 'Snacks'
])

    vehicle_type = st.sidebar.selectbox('Type of Vehicle', 
        ['motorcycle', 'scooter', 'bicycle', 'van'])

    # festival = st.sidebar.selectbox('Is it a festival?', ['No', 'Yes'])
    city = st.sidebar.selectbox('City Type', ['Urban', 'Metropolitian', 'Semi-Urban', 'Other'])

    # Create a dictionary from the inputs
    data = {
        'Delivery_person_Age': age,
        'Delivery_person_Ratings': ratings,
        # 'Vehicle_condition': 2 if vehicle_condition == 'Excellent' else 1 if vehicle_condition == 'Good' else 0,
        'Distance_km': distance,
        # One-hot encoded features will be created from these
        'Weatherconditions': weather,
        'Road_traffic_density': traffic,
        'Type_of_order': order_type,
        'Type_of_vehicle': vehicle_type,
        # 'Festival': festival,
        'Store_latitude': store_latitude,
        'Store_longitude': store_longitude,
        'Delivery_location_latitude': drop_latitude,
        'Delivery_location_longitude': drop_longitude,
        'City': city
    }
    
    # Convert dictionary to a pandas DataFrame
    features_df = pd.DataFrame(data, index=[0])
    return features_df

input_df = get_user_inputs()

# --- Prediction Logic ---
if st.sidebar.button('Predict Delivery Time'):
    # One-hot encode the categorical features
    # This step converts text categories into a numerical format for the model
    input_encoded = pd.get_dummies(input_df, dtype=int)
    
    # Align columns: Ensure the input has the exact same columns as the training data
    # Add any missing columns and fill them with 0
    final_input = pd.DataFrame(columns=model_columns)
    final_input = pd.concat([final_input, input_encoded], ignore_index=True, sort=False).fillna(0)
    
    # Ensure the column order is identical to the training data
    final_input = final_input[model_columns]

    # Make the prediction
    prediction = model.predict(final_input)
    
    # Display the result
    st.subheader("Predicted Delivery Time")
    prediction_minutes = round(prediction[0])
    st.success(f"**The estimated delivery time is approximately {prediction_minutes} minutes.**")

    with st.expander("See Input Details"):
        st.write(input_df)
else:
    st.info("Enter the delivery details in the sidebar and click the 'Predict' button to get an estimate.")

