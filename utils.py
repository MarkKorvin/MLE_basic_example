import os
import logging

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

def get_project_dir(sub_dir: str) -> str:
    """Return path to a project subdirectory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), sub_dir))

def configure_logging() -> None:
    """Configures logging"""
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')