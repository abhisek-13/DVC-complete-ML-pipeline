import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logger object
logger = logging.getLogger("feature_engineering")
logger.setLevel('DEBUG')

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')

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
      
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: str) -> pd.DataFrame:
    """
    This function applies TF-IDF vectorization to the specified text column of the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        text_column (str): The name of the column containing the text data.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test
        
        logger.debug("TF-IDF vectorization applied to train and test data.")
        
        return train_df, test_df
    except Exception as e:
        logger.error("Error occured during TF-IDF vectorization: %s", e)
        raise
      
def save_data(df: pd.DataFrame, file_path: str) -> None:
  """
  This function saves the train data to CSV files in the specified directory.
  Args:
      train_data (pd.DataFrame): _description_
      file_path (str): _description_
  """
  try:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.debug("Data saved to %s", file_path)
  except Exception as e:
    logger.error("Error occurred while saving the data: %s", e)
    raise



def main():
    """
    This function is the main entry point for the script. It loads the dataset, preprocesses it, and saves the preprocessed DataFrame.
    """
    try:
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']
        
        train_data = load_data("data/interim/train_processed.csv")
        test_data = load_data('data/interim/test_processed.csv')
        
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)
        
        save_data(train_df, os.path.join('data', 'processed', 'train_tfidf.csv'))
        save_data(test_df, os.path.join('data', 'processed', 'test_tfidf.csv'))
        
        logger.debug("Feature engineering completed and data saved.")
        
    except Exception as e:
        logger.error("An error occurred during the preprocessing: %s", e)
        raise
      
if __name__ == "__main__":
    main()