import pandas as pd
from data_preprocessing import create_data_pipeline, save_pipeline, load_pipeline, split_data, encode_response_variable
from ml_functions import training_pipeline, prediction_pipeline, evaluation_matrices
from helper_functions import logging

def main():
    # Configure logging (optional)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the dataset (using your uploaded CSV)
    df = pd.read_csv('"C:\Users\Admin\OneDrive\Desktop\data.csv')
    logging.info('Data loaded successfully.')

    # Feature engineering: separate features (X) and target (y)
    X = df.drop(['Class'], axis=1)  # 'Class' is the target variable
    y = df['Class']  # Target column is 'Class'

    # Encode response variable (assuming encode_response_variable is defined)
    y_encoded = encode_response_variable(y)

    # Create and fit the data processing pipeline (replace with create_data_pipeline)
    pipeline = create_data_pipeline(X)
    pipeline.fit(X)
    logging.info('Data processing pipeline created and fitted.')

    # Save the pipeline for later use (assuming save_pipeline is defined)
    save_pipeline(pipeline, 'D:/shrilaxmi/data/data_pipeline.pkl')
    logging.info('Data processing pipeline saved.')

    # Transform the data using the fit_transform method
    X_transformed = pipeline.transform(X)

    # Split the data for training and validation
    X_train, X_val, y_train, y_val = split_data(X_transformed, y_encoded)

    # Train the best model (replace with training_pipeline)
    best_model = training_pipeline(X_train, y_train)

    # Make predictions (replace with prediction_pipeline)
    predictions = prediction_pipeline(best_model, X_val)

    # Evaluate the model (replace with evaluation_matrices)
    conf_matrix, acc_score, class_report = evaluation_matrices(y_val, predictions)

    logging.info('Model training, prediction, and evaluation completed.')
    logging.info(f'Confusion Matrix:\n{conf_matrix}')
    logging.info(f'Accuracy Score: {acc_score}')
    logging.info(f'Classification Report:\n{class_report}')

if __name__ == "__main__":
    main()
