import streamlit as st
import pandas as pd
import pickle
import requests
import os
from PIL import Image
import logging

# Configure logging to write to a file in the current directory
logging.basicConfig(
    filename='logs/logfile_UI.txt',  
    level=logging.DEBUG,      
    format='%(asctime)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Function to load artifacts
def load_artifact(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError as e:
        logging.error(f"Artifact file not found: {filename}")
        raise

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Adhoc Risk Profiling", "Batch Profiling"])

# Determine the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the images folder
images_dir = os.path.join(script_dir, '..', 'images')

# Layout: Image on the left, title on the right
col1, col2 = st.columns([1, 3])
with col1:
    image_path = os.path.join(images_dir, 'risk-image2.jfif')
    image = Image.open(image_path)
    st.image(image, use_column_width=True)

with col2:
    st.title("Loan Risk Categorization")
    image_path = os.path.join(images_dir, 'risk-image.png')
    image = Image.open(image_path)

# Navigation logic
if page == "Home":
    st.write("Welcome to the Loan Risk Categorization App.")
    st.write("Use the sidebar to navigate to Adhoc or Batch Profiling.")
elif page == "Adhoc Risk Profiling":
    st.header("Enter customer details:")
    
    # Adjusted inputs for the dataset features
    I0 = st.number_input("I0 (Initial Measurement)", min_value=0.0)
    PA500 = st.number_input("PA500 (Power at 500Hz)", min_value=0.0)
    HFS = st.number_input("HFS (High-Frequency Stability)", min_value=0.0)
    DA = st.number_input("DA (Damping Amplitude)", min_value=0.0)
    Area = st.number_input("Area", min_value=0.0)
    A_DA = st.number_input("A.DA (Area Damping Amplitude)", min_value=0.0)
    Max_IP = st.number_input("Max.IP (Max Intensity Peak)", min_value=0.0)
    DR = st.number_input("DR (Damping Ratio)", min_value=0.0)
    P = st.number_input("P (Power)", min_value=0.0)

    if st.button('Predict Risk Category'):
        # Load necessary artifacts
        pipeline = load_artifact(os.path.join(script_dir, '..', 'artifacts', 'data_processing_pipeline.pkl'))
        model = load_artifact(os.path.join(script_dir, '..', 'artifacts', 'best_classifier.pkl'))
        label_encoder = load_artifact(os.path.join(script_dir, '..', 'artifacts', 'label_encoder.pkl'))

        # Create DataFrame with input data
        input_df = pd.DataFrame([[I0, PA500, HFS, DA, Area, A_DA, Max_IP, DR, P]], 
                                columns=['I0', 'PA500', 'HFS', 'DA', 'Area', 'A.DA', 'Max.IP', 'DR', 'P'])
        logging.info(f"User input data frame created: {input_df}")

        # Transform input using the pipeline
        transformed_input = pipeline.transform(input_df)
        prediction = model.predict(transformed_input)
        decoded_prediction = label_encoder.inverse_transform(prediction)

        # Display prediction
        st.subheader('Predicted Risk Category:')
        st.write(decoded_prediction[0])
        logging.info(f"Prediction: {decoded_prediction[0]}")

elif page == "Batch Profiling":
    st.header("Batch Profiling")
    uploaded_file = st.file_uploader("Upload your CSV file for batch prediction", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        logging.info(f"Batch file uploaded with {len(df)} records")

        try:
            # Send the data for batch prediction
            response = requests.post("http://fastapi-container:8001/batch_predict", json={"data": df.to_dict(orient="list")})
            
            response.raise_for_status()
            predictions = response.json()
            output_df = pd.DataFrame(predictions)
            
            # Save batch predictions
            output_folder = os.path.join(script_dir, '..', 'Data', 'output')
            os.makedirs(output_folder, exist_ok=True)
            output_file_path = os.path.join(output_folder, 'batch_predictions.csv')
            output_df.to_csv(output_file_path, index=False)
            
            st.success(f"Batch predictions saved to {output_file_path}")
            logging.info(f"Batch predictions saved to {output_file_path}")
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error occurred during batch prediction: {http_err}")
            logging.error(f"Batch prediction failed due to HTTP error: {http_err}")
        except requests.exceptions.RequestException as req_err:
            st.error("Error during batch prediction. Please check the API service.")
            logging.error(f"Batch prediction failed: {req_err}")
