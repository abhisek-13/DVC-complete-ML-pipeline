import logging
import functools
import inspect

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

def log_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cls_name = args[0].__class__.__name__ if args else ""
        method_name = func.__name__
        try:
            logging.info(f"{cls_name}.{method_name} - STARTED")
            result = func(*args, **kwargs)
            logging.info(f"{cls_name}.{method_name} - COMPLETED")
            return result
        except Exception as e:
            logging.error(f"{cls_name}.{method_name} - ERROR: {str(e)}")
            raise
    return wrapper

# Sample class using the logger
class MyService:
    @log_decorator
    def do_something(self, x, y):
        return x / y

    @log_decorator
    def risky_method(self):
        raise ValueError("Something went wrong!")

# Example usage
if __name__ == "__main__":
    service = MyService()
    service.do_something(10, 2)
    
    try:
        service.risky_method()
    except Exception:
        pass
