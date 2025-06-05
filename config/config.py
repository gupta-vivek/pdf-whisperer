import json
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# Get the directory where config.py is located
CONFIG_DIR = Path(__file__).parent

with open(CONFIG_DIR / 'config.json', 'r') as file:
    config = json.load(file)


# Accessing configuration values
class Config:
    config_dir = CONFIG_DIR
    api_key = os.environ['DEEPSEEK_API_KEY']
    model = config['model']
    base_url = config['base_url']
    prompt_template = config['prompt_template']
    lance_db_uri = config['lance_db_uri']
    lance_db_table = config['lance_db_table']
