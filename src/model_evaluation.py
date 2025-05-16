import os
import json
import numpy as np
import pandas as pd
import logging
import pickle
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score)
from dvclive import Live
import yaml

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logger object
logger = logging.getLogger("model_evaluation")
logger.setLevel('DEBUG')

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')

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

def load_model(model_path: str):
    """
    This function loads the model from the given model_path.
    Args:
        model_path (str): The path to the model file.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully: %s", model_path)
        return model
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred during loading the model: %s", e)
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
      
      
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    This function evaluates the model on the test data and returns the evaluation metrics.
    Args:
        model: The trained model.
        X_test (np.ndarray): The test data.
        y_test (np.ndarray): The true labels for the test data.
    """
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }
        
        logger.debug("Model evaluation metrics: %s", metrics)
        return metrics
    except Exception as e:
        logger.error("Unexpected error occurred during model evaluation: %s", e)
        raise
      
def save_metrics(metrics: dict, file_path: str) -> None:
    """
    This function saves the evaluation metrics to a JSON file.
    Args:
        metrics (dict): The evaluation metrics.
        file_path (str): The path to save the metrics file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
      
        with open(file_path, 'w') as f:
            json.dump(metrics, f,indent=4)
        logger.debug("Metrics saved successfully: %s", file_path)
    except Exception as e:
        logger.error("Unexpected error occurred during saving the metrics: %s", e)
        raise
      
def main():
  """
  Main function to load the model, data, evaluate the model and save the metrics.
  """
  try:
      
    params = load_params(params_path='params.yaml')
    model = load_model("models/model.pkl")
    
    test_data = load_data("data/processed/test_tfidf.csv")
    
    X_test = test_data.iloc[:,:-1].values
    y_test = test_data.iloc[:,-1].values
    
    metrics = evaluate_model(model, X_test, y_test)
    
    # Experiment tracking using DVC Live
    with Live(save_dvc_exp=True) as live:
        live.log_metric('accuracy', accuracy_score(y_test,y_test))
        live.log_metric('precision', precision_score(y_test,y_test))
        live.log_metric('recall', recall_score(y_test,y_test))
        
        live.log_params(params)
    
    
    save_metrics(metrics, "reports/metrics.json")
    logger.debug("Model evaluation completed successfully")
  except Exception as e:
    logger.error("An error occurred during Model evaluation: %s", e)
    raise
  
if __name__ == "__main__":
  main()