import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

# Navigate to the main project directory
main_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Ensure the logs directory exists in the main project directory
log_dir = os.path.join(main_project_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'logfile_API.txt'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load artifacts
def load_artifact(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError as e:
        logging.error(f"Artifact file not found: {filename}")
        raise

# Adjust the artifacts path to the main project directory
artifacts_path = os.path.join(main_project_dir, 'artifacts')
data_processing_pipeline = load_artifact(os.path.join(artifacts_path, 'data_processing_pipeline.pkl'))
best_classifier = load_artifact(os.path.join(artifacts_path, 'best_classifier.pkl'))
label_encoder = load_artifact(os.path.join(artifacts_path, 'label_encoder.pkl'))

app = FastAPI()

# Root route to handle the root path
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Batch Prediction API"}

class BatchRequest(BaseModel):
    data: dict

@app.post("/batch_predict")
async def batch_predict(request: BatchRequest):
    try:
        df = pd.DataFrame.from_dict(request.data)
        logging.info(f"Received batch request with {len(df)} records.")
        
        if df.empty:
            logging.error("Received empty DataFrame.")
            raise HTTPException(status_code=400, detail="Received empty DataFrame.")
        
        # Preserve the CustomerID
        if 'CustomerID' not in df.columns:
            logging.error("CustomerID column is missing.")
            raise HTTPException(status_code=400, detail="CustomerID column is missing.")
        
        customer_ids = df['CustomerID']  # Extract CustomerID for later use
        df = df.drop(columns=['CustomerID'])  # Drop CustomerID before processing
        
        # Transform the input data
        transformed_input = data_processing_pipeline.transform(df)
        logging.info(f"Batch data transformed successfully")
        
        # Predict
        predictions = best_classifier.predict(transformed_input)
        decoded_predictions = label_encoder.inverse_transform(predictions)
        logging.info(f"Batch predictions completed")
        
        if not len(decoded_predictions):
            logging.error("Predictions are empty.")
            raise HTTPException(status_code=500, detail="Predictions are empty.")
        
        # Combine CustomerID with predictions
        result_df = pd.DataFrame({
            'CustomerID': customer_ids,
            'Predicted Risk Category': decoded_predictions
        })
        
        # Save predictions to CSV in the main project directory
        output_folder = os.path.join(main_project_dir, 'Data', 'output')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'batch_predictions.csv')
        result_df.to_csv(output_path, index=False)
        logging.info(f"Batch predictions saved to {output_path}")
        
        # Return predictions as JSON
        return result_df.to_dict(orient='records')
        
    except Exception as e:
        logging.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
