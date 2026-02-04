import yaml
from pymongo import MongoClient
from urllib.parse import quote_plus
from db_connect import db_connect

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Connect to MongoDB
collection = db_connect()

# Delete all documents
result = collection.delete_many({})
print(f"Deleted {result.deleted_count} documents from {config['mongodb_collection']}")
