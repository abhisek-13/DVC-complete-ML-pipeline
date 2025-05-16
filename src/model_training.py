import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logger object
logger = logging.getLogger("model_training")
logger.setLevel('DEBUG')

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')

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

def loag_params(params_path: str) -> dict:
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

def load_data(data_path: str) -> pd.DataFrame:
    """
    This function loads the data from the given data_path and returns a pandas DataFrame.
    Args:
        data_url (str): The URL of the CSV file to be loaded.
    """
    try:
        df = pd.read_csv(data_path)
        df.fillna('', inplace=True)
        logger.debug("Data loaded and NaN values removed: %s", data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred during loading the data: %s", e)
        raise
      
def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    This function trains a Random Forest Classifier on the given training data and returns the trained model.
    Args:
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The target labels.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
          raise ValueError("The number of samples in X_train and y_train must be equal.")
          logger.debug("Shape of X_train and y_train is not same so training failed")
       
        logger.debug("Initiating Randomforest model with parameters: ",params)
        model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        logger.debug("Training the model with data of size: %s", X_train.shape[0])
        model.fit(X_train, y_train)
        
        logger.debug("Model trained successfully")
        return model
    except ValueError as e:
        logger.error("ValueError during model training: %s", e)
        raise
    except Exception as e:
        logger.error("Error occurred during model training: %s", e)
        raise
      
      
def save_model(model, file_path: str) -> None:
  """
  This function saves the trained model to the specified file path using pickle.
  Args:
      model (_type_): _description_
      file_path (str): _description_
  """
  
  try:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f:
      pickle.dump(model, f)
    
    logger.debug("Model saved to: %s", file_path)
    
  except FileNotFoundError as e:
    logger.error("File path not found: %s", e)
    raise
  except Exception as e:
    logger.error("Error occurred while saving the model: %s", e)
    raise
  
def main():
    """
    Main function to execute the model training process.
    It loads the data from a CSV file, trains a Random Forest Classifier, and saves the trained model.
    """
    try:
        params = loag_params(params_path='params.yaml')['model_training']
        
        train_data = load_data('data/processed/train_tfidf.csv')
        
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        
        model = train_model(X_train, y_train, params)


        model_save_path = 'models/model.pkl'
        save_model(model, model_save_path)
        
    except Exception as e:
        logger.error("An error occurred during Model building: %s", e)
        raise
      
if __name__ == "__main__":
    main()