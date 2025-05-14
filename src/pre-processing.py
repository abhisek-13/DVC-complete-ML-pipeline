import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logger object
logger = logging.getLogger("data_preprocessing")
logger.setLevel('DEBUG')

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')

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


def transform_text(text:str) -> str:
    """
    This function transforms the text by removing punctuation, converting to lowercase, and stemming.
    Args:
        text (str): The text to be transformed.
    """
    ps = PorterStemmer()
    
    # convert to lowercase
    text = text.lower()
    
    # tokenize the text
    text = nltk.word_tokenize(text)
    
    # Remove non-alphabetic characters
    text = [word for word in text if word.isalpha()]
    
    # Remove stop words and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # stemming
    text = [ps.stem(word) for word in text]
    
    # join the words back into a single string
    text = ' '.join(text)
    
    return text
  
def preprocess_df(df: pd.DataFrame, text_column = 'text', target_column = 'target') -> pd.DataFrame:
    """
    This function preprocesses the DataFrame by transforming the text and encoding the target variable.
    Args:
        df (pd.DataFrame): The DataFrame to be preprocessed.
        text_column (str): The name of the text column.
        target_column (str): The name of the target column.
    """
    try:
      logger.debug("Starting data preprocessing...")
      
      # Encode the target column
      encode = LabelEncoder()
      df[target_column] = encode.fit_transform(df[target_column])
      logger.debug("Target column encoded.")
      
      # removing duplicate rows
      df = df.drop_duplicates()
      logger.debug("Duplicate rows removed.")
      
      # removing null values
      # df = df.dropna(subset=[text_column, target_column])
      
      print(type(df))
      # applying transform_text function to the text column
      df[text_column] = df[text_column].apply(transform_text)
      logger.debug("Text column transformed.")
      
      
      logger.debug("Data preprocessing completed.")
      return df
  
    except KeyError as e:
      logger.error("Column not found: %s", e)
      raise
    except Exception as e:
      logger.error("Error occured during preprocessing: %s", e)
      raise
  
def main(text_column = 'text', target_column = 'target'):
    """
    This function is the main entry point for the script. It loads the dataset, preprocesses it, and saves the preprocessed DataFrame.

    Args:
        text_column (str, optional): _description_. Defaults to 'text'.
        target_column (str, optional): _description_. Defaults to 'target'.
    """
    
    try:
        # fetching the data from the data/raw
        train_data = pd.read_csv("data\\raw\\train.csv")
        test_data = pd.read_csv('data\\raw\\test.csv')
        
        # Transforming the data
        train_processed_data = preprocess_df(train_data,text_column,target_column)
        test_preprocessed_data = preprocess_df(test_data,text_column,target_column)
        
        # storing the preprocessed data in data/interim
        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_preprocessed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        
        logger.debug("Preprocessed data saved at %s:", data_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("Empty data error: %s", e)
        raise
    except Exception as e:
        logger.error("An error occurred during the preprocessing: %s", e)
        raise
    


if __name__ == "__main__":
    main()