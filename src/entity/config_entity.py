from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    db_uri: str
    raw_data_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE : str
    data_path: Path
    all_schema: dict
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path : Path
    preprocessor_obj_file_path : str
    target_column: str
    

@dataclass(frozen=True)
class ModeltrainerConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    params: dict
    target_column : str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    target_column : str    
    
    