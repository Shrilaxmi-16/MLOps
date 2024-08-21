import os
import pandas as pd
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

# Navigate to the main project directory
main_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load your datasets
reference_data = pd.read_csv(os.path.join(main_project_dir, 'Data/Banking_Credit_Risk_Data.csv'))
current_data = pd.read_csv(os.path.join(main_project_dir, 'Data/test6.csv'))

# Ensure CustomerID is of the same type in both datasets (convert to string)
reference_data['CustomerID'] = reference_data['CustomerID'].astype(str)
current_data['CustomerID'] = current_data['CustomerID'].astype(str)

# Remove the 'RiskCategory' and 'CustomerID' columns from the reference dataset for drift analysis
columns_to_exclude = ['RiskCategory', 'CustomerID']
reference_data = reference_data.drop(columns=[col for col in columns_to_exclude if col in reference_data.columns])
current_data = current_data.drop(columns=[col for col in columns_to_exclude if col in current_data.columns])

# Create a list of ColumnDriftMetrics for each column (excluding 'RiskCategory' and 'CustomerID')
drift_metrics = [ColumnDriftMetric(column_name=col) for col in reference_data.columns if col in current_data.columns]

# Create a report with drift metrics for each column
report = Report(metrics=[
    DatasetDriftMetric(),  # Detects overall dataset drift
    *drift_metrics,        # Detects drift for each individual column
])

# Create the 'report' folder in the main project directory if it doesn't exist
report_folder = os.path.join(main_project_dir, 'report')
os.makedirs(report_folder, exist_ok=True)

# Generate the report and save it in the 'report' folder
report_path = os.path.join(report_folder, 'evidently_report.html')
report.run(current_data=current_data, reference_data=reference_data)
report.save_html(report_path)

print(f"Evidently report generated and saved as '{report_path}'.")
