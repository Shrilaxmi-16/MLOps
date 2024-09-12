import pandas as pd
from data_preprocessing import create_data_pipeline, save_pipeline, load_pipeline, split_data, encode_response_variable
from ml_functions import training_pipeline, prediction_pipeline, evaluation_matrices
from helper_functions import logging


def main():
    # Configure logging (optional, adjust log level and output destination as needed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data/Banking_Credit_Risk_Data.csv
    df = pd.read_csv('C:/Users/Admin/Desktop/Basudev/DSC/MLOPs/08-08/data/Banking_Credit_Risk_Data.csv')
    logging.info('Data loaded successfully.')

    # Feature engineering (replace with your feature engineering steps)
    X = df.drop(['CustomerID', 'RiskCategory'], axis=1)
    y = df['RiskCategory']

    # Encode response variable (assuming encode_response_variable is defined)
    y_encoded = encode_response_variable(y)

    # Create and fit the data processing pipeline (replace with create_data_pipeline)
    pipeline = create_data_pipeline(X)
    pipeline.fit(X)
    logging.info('Data processing pipeline created and fitted.')

    # Save the pipeline for later use (assuming save_pipeline is defined)
    save_pipeline(pipeline, 'C:\Users\Admin\OneDrive\Desktop\data.csv')
    logging.info('Data processing pipeline saved.')

    # Transform the data using the fit_transform method
    X_transformed = pipeline.transform(X)

    # Split the data for training and validation
    X_train, X_val, y_train, y_val = split_data(X_transformed, y_encoded)

    # Train the best model (replace with training_pipeline)
    best_model = training_pipeline(X_train, y_train)

    # Make predictions (replace with prediction_pipeline)
    predictions = prediction_pipeline(X_val)

    # Evaluate the model (replace with evaluation_matrices)
    conf_matrix, acc_score, class_report = evaluation_matrices(X_val, y_val)

    logging.info('Model training, prediction, and evaluation completed.')


if __name__ == "__main__":
    main()
