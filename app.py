import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open("fetal-health-adasyn.pkl", 'rb') as file:
    model = pickle.load(file)

# Set page configuration
st.set_page_config(page_title="Fetal Health Classification", page_icon="🍼", layout="wide")

# Custom CSS for styling and layout
st.markdown("""
    <style>
        .stApp {
            background-color: #f7f9fc;
        }
        .sidebar .sidebar-content {
            padding: 10px;
        }
        .main-content {
            max-width: 800px;
            margin: auto;
            padding: 20px;
        }
        .stTitle h1 {
            font-family: 'Arial', sans-serif;
            color: #4A4A4A;
            font-size: 2.5em;
            font-weight: 600;
            text-align: center;
        }
        .stButton>button {
            background-color: #ff7f50;
            color: white;
            border: none;
            padding: 10px 25px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #e67342;
        }
        .stNumberInput>div>div>input {
            border: 2px solid #ff7f50;
            border-radius: 5px;
            padding: 5px;
        }
        .prediction-box {
            background-color: black;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5em;
        }
        .suggestion {
            font-family: 'Arial', sans-serif;
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
        }
        .success {
            background-color: #28a745;
            color: white;
        }
        .warning {
            background-color: #ffc107;
            color: white;
        }
        .error {
            background-color: #dc3545;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for input fields
st.sidebar.header("Enter Features for Prediction")

# Input fields for each feature with tooltips
baseline_value = st.sidebar.number_input("Baseline Value", value=120.0, format="%.2f", help="The baseline fetal heart rate in beats per minute (bpm).")
accelerations = st.sidebar.number_input("Accelerations", value=0.0, format="%.4f", help="Number of accelerations observed.")
fetal_movement = st.sidebar.number_input("Fetal Movement", value=0.0, format="%.4f", help="Frequency of fetal movements observed.")
uterine_contractions = st.sidebar.number_input("Uterine Contractions", value=0.0, format="%.4f", help="Number of uterine contractions observed.")
light_decelerations = st.sidebar.number_input("Light Decelerations", value=0.0, format="%.4f", help="Number of light decelerations observed.")
severe_decelerations = st.sidebar.number_input("Severe Decelerations", value=0.0, format="%.4f", help="Number of severe decelerations observed.")
prolongued_decelerations = st.sidebar.number_input("Prolongued Decelerations", value=0.0, format="%.4f", help="Number of prolonged decelerations observed.")
abnormal_short_term_variability = st.sidebar.number_input("Abnormal Short Term Variability", value=73.0, format="%.2f", help="Abnormal short term variability measurement.")
mean_value_of_short_term_variability = st.sidebar.number_input("Mean Value of Short Term Variability", value=0.5, format="%.2f", help="Mean value of short term variability.")
percentage_of_time_with_abnormal_long_term_variability = st.sidebar.number_input("Percentage of Time with Abnormal Long Term Variability", value=43.0, format="%.2f", help="Percentage of time with abnormal long term variability.")
mean_value_of_long_term_variability = st.sidebar.number_input("Mean Value of Long Term Variability", value=0.0, format="%.2f", help="Mean value of long term variability.")
histogram_width = st.sidebar.number_input("Histogram Width", value=0.0, format="%.2f", help="Width of the histogram.")
histogram_min = st.sidebar.number_input("Histogram Min", value=62.0, format="%.2f", help="Minimum value in the histogram.")
histogram_max = st.sidebar.number_input("Histogram Max", value=126.0, format="%.2f", help="Maximum value in the histogram.")
histogram_number_of_peaks = st.sidebar.number_input("Histogram Number of Peaks", value=2.0, format="%.2f", help="Number of peaks in the histogram.")
histogram_number_of_zeroes = st.sidebar.number_input("Histogram Number of Zeroes", value=0.0, format="%.2f", help="Number of zeroes in the histogram.")
histogram_mode = st.sidebar.number_input("Histogram Mode", value=120.0, format="%.2f", help="Mode of the histogram.")
histogram_mean = st.sidebar.number_input("Histogram Mean", value=137.0, format="%.2f", help="Mean of the histogram.")
histogram_median = st.sidebar.number_input("Histogram Median", value=121.0, format="%.2f", help="Median of the histogram.")
histogram_variance = st.sidebar.number_input("Histogram Variance", value=73.0, format="%.2f", help="Variance of the histogram.")
histogram_tendency = st.sidebar.number_input("Histogram Tendency", value=1.0, format="%.2f", help="Tendency of the histogram.")

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])  # Creates three columns with center column larger

with col2:
    st.title("Fetal Health Classification")

    # Add the image from the provided link
    st.image("https://cdn-icons-png.freepik.com/256/8034/8034964.png", caption="Fetal Health Classification Dataset", width=400, use_column_width=False)

    # Create a dictionary for the input features
    new_data = {
        'baseline value': baseline_value,
        'accelerations': accelerations,
        'fetal_movement': fetal_movement,
        'uterine_contractions': uterine_contractions,
        'light_decelerations': light_decelerations,
        'severe_decelerations': severe_decelerations,
        'prolongued_decelerations': prolongued_decelerations,
        'abnormal_short_term_variability': abnormal_short_term_variability,
        'mean_value_of_short_term_variability': mean_value_of_short_term_variability,
        'percentage_of_time_with_abnormal_long_term_variability': percentage_of_time_with_abnormal_long_term_variability,
        'mean_value_of_long_term_variability': mean_value_of_long_term_variability,
        'histogram_width': histogram_width,
        'histogram_min': histogram_min,
        'histogram_max': histogram_max,
        'histogram_number_of_peaks': histogram_number_of_peaks,
        'histogram_number_of_zeroes': histogram_number_of_zeroes,
        'histogram_mode': histogram_mode,
        'histogram_mean': histogram_mean,
        'histogram_median': histogram_median,
        'histogram_variance': histogram_variance,
        'histogram_tendency': histogram_tendency
    }

    # Convert to DataFrame
    new_data_df = pd.DataFrame([new_data])

    # Prediction button
    if st.button('Predict Fetal Health Status'):
        # Make the prediction
        prediction = model.predict(new_data_df)
        
        # Handle the prediction value
        try:
            # Convert prediction to float first and then to int
            prediction_value = float(prediction[0])
            prediction_int = int(round(prediction_value))  # Round to nearest integer
            
            # Map prediction to health status
            health_status = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
            status = health_status.get(prediction_int, "Unknown")
            
            # Display prediction with black background and white text
            st.markdown(f"<div class='prediction-box'>Predicted Fetal Health Status: {status}</div>", unsafe_allow_html=True)
            
            # Provide suggestions based on the prediction
            if prediction_int == 1:
                st.markdown("<div class='suggestion success'>The fetal health status is Normal. Continue regular check-ups.</div>", unsafe_allow_html=True)
            elif prediction_int == 2:
                st.markdown("<div class='suggestion warning'>The fetal health status is Suspect. Please consult your healthcare provider for further evaluation.</div>", unsafe_allow_html=True)
            elif prediction_int == 3:
                st.markdown("<div class='suggestion error'>The fetal health status is Pathological. Immediate consultation with your healthcare provider is recommended.</div>", unsafe_allow_html=True)
        except ValueError as e:
            st.write(f"Error in prediction conversion: {e}")
        except KeyError as e:
            st.write(f"Prediction not found in health status dictionary: {e}")
