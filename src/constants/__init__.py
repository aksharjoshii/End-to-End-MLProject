from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params/params.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")   
MONGO_DB_URL = "mongodb+srv://akshar1895:Aksharsdata@cluster0.bdyjsdd.mongodb.net/?retryWrites=true&w=majority"
MONGO_DATABASE_NAME = "automatidata"
MONGO_COLLECTION_NAME = "nyctaxi"         

PREPROCESSOR_OBJ_PATH = "artifacts/data_transformation/preprocessor.joblib"
MODEL_OBJ_PATH = "artifacts/model_trainer/clf_model.joblib"
