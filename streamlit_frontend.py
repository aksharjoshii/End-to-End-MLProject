import streamlit as st
import requests
# Set page configuration options
st.set_page_config(
    page_title="My Streamlit App",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="collapsed"
    )

st.header(":red[NYC Taxi Tip Prediction]", divider='rainbow')

# Sidebar to toggle between Train and Predict modes
mode = st.sidebar.radio("Select Mode", ("Train", "Predict"))
if mode == "Predict":
    # Streamlit Input Form
    st.subheader("Input Parameters")

    # Input columns
    col1, col2 = st.columns(2)

    vendor_id = col1.number_input("VendorID (1 or 2)", min_value=1, max_value=2, step=1, value=1)
    pickup_datetime = col1.text_input("Pickup datetime", "08/17/2017 4:06:26 AM")
    dropoff_datetime = col1.text_input("Dropoff datetime", "08/17/2017 4:06:29 AM")
    passenger_count = col1.number_input("Number of passengers", min_value=0, max_value=6, step=1, value=2)
    rate_code_id = col1.number_input("Rate code ID (1-5)", min_value=1, max_value=5, step=1, value=1)
    pickup_location_id = col1.number_input("Pickup Location ID (1-265)", min_value=1, max_value=265, step=1, value=101)
    dropoff_location_id = col1.number_input("Dropoff Location ID (1-265)", min_value=1, max_value=265, step=1, value=102)
    mean_duration = col2.number_input("Mean duration of trips", value=30.5)
    mean_distance = col2.number_input("Mean distance of trips", value=5.0)
    predicted_fare = col2.number_input("Predicted fare", value=20.0)

    if st.button("Predict"):
        # Call FastAPI endpoint for prediction
        api_url = "http://localhost:8000/predict"
        payload = {
            "VendorID": vendor_id,
            "tpep_pickup_datetime": pickup_datetime,
            "tpep_dropoff_datetime": dropoff_datetime,
            "passenger_count": passenger_count,
            "RatecodeID": rate_code_id,
            "PULocationID": pickup_location_id,
            "DOLocationID": dropoff_location_id,
            "mean_duration": mean_duration,
            "mean_distance": mean_distance,
            "predicted_fare": predicted_fare
        }
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            prediction_result = response.json()
            st.success(f"Prediction: {prediction_result['prediction']}")
        else:
            st.error("Error making prediction. Please check the input parameters.")
elif mode=='Train':
    # Streamlit Training Button
    st.subheader("Model Training")
    if st.button("Train Model"):
        # Call FastAPI endpoint for training
        train_api_url = "http://localhost:8000/train"
        train_response = requests.get(train_api_url)

        if train_response.status_code == 200:
            st.success("Model training successful!")
        else:
            st.error("Error during model training.")


