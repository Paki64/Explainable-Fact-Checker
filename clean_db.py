import yaml
from pymongo import MongoClient
from urllib.parse import quote_plus

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Connect to MongoDB
uri = config["mongodb_uri"]
username = config.get("mongodb_username", "")
password = config.get("mongodb_password", "")
auth_db = config.get("mongodb_auth_db", "admin")

if username and password:
    escaped_username = quote_plus(username)
    escaped_password = quote_plus(password)
    uri = uri.replace("mongodb://", f"mongodb://{escaped_username}:{escaped_password}@")
    client = MongoClient(uri, authSource=auth_db, serverSelectionTimeoutMS=5000)
else:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)

db = client[config["mongodb_db"]]
collection = db[config["mongodb_collection"]]

# Delete all documents
result = collection.delete_many({})
print(f"Deleted {result.deleted_count} documents from {config['mongodb_collection']}")
