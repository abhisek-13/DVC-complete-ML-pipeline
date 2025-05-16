import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml


# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logger object
logger = logging.getLogger("data_ingestion")
logger.setLevel('DEBUG')

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')

# file handler
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# adding formatter to console and file handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
  """Loads the parameters from the given YAML file.
  Args:
    params_path (str): The path to the YAML file containing parameters.
  """
  
  try:
    with open(params_path, 'r') as file:
      params = yaml.safe_load(file)
    logger.debug("Parameters loaded from: %s", params_path)
    return params
  except FileNotFoundError as e:
    logger.error("Parameters file not found: %s", e)
    raise
  except yaml.YAMLError as e:
    logger.error("Error parsing YAML file: %s", e)
    raise
  except Exception as e:
    logger.error("Unexpected error occurred during loading parameters: %s", e)
    raise



def load_data(data_url: str) -> pd.DataFrame:
  """
  This function loads the data from the given URL and returns a pandas DataFrame.
  Args:
    data_url (str): The URL of the CSV file to be loaded.
  """
  try:
    df = pd.read_csv(data_url)
    logger.debug("Data loaded from url: %s", data_url)
    return df
  except pd.errors.ParserError as e:
    logger.error("Error parsing CSV file: %s", e)
    raise
  except Exception as e:
    logger.error("Unexpected error occured duing loading the data: %s", e)
    raise
  
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
  """
  This function preprocesses the data by dropping unnecessary columns and renaming the remaining columns.
  Args:
    df (pd.DataFrame): The DataFrame to be preprocessed.
  """
  try:
    df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"], inplace=True)
    df.rename(columns={'v1': 'target','v2':'text'}, inplace=True)
    logger.debug("Data preprocessing completed.")
    return df
  except KeyError as e:
    logger.error("Missing columns in the dataframe: %s", e)
    raise
  except Exception as e:
    logger.error("Unexpected error occured during preprocessing: %s", e)
    raise
  
def save_data(train_data: pd.DataFrame,test_data:pd.DataFrame, data_path: str) -> None:
  """
  This function saves the train and test data to CSV files in the specified directory.
  Args:
    train_data (pd.DataFrame): The training data to be saved.
    test_data (pd.DataFrame): The testing data to be saved.
    data_path (str): The directory where the CSV files will be saved.
  """
  try:
    raw_data_path = os.path.join(data_path, "raw")
    os.makedirs(raw_data_path, exist_ok=True)
    train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
    logger.debug("Train and Test data saved to: %s", data_path)
  except Exception as e:
    logger.error("Unexpected error occured during saving data to CSV: %s", e)
    raise
  
  
def main():
  """
  Main function to execute the data ingestion process.
  It loads the data from a URL, preprocesses it, splits it into training and testing sets,
  """
  try:
    params = load_params(params_path="params.yaml")
    test_size = params['data_ingestion']['test_size']
    
    data_url = "https://raw.githubusercontent.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/refs/heads/main/experiments/spam.csv"
    
    df = load_data(data_url)
    
    final_df = preprocess_data(df)
    
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    save_data(train_data, test_data, data_path="./data")
  
  except Exception as e:
    logger.error("Failed to complete the data ingestion process: %s", e)
    print(f"error: {e}")
    
    
if __name__ == "__main__":
  main()
  
  