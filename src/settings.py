from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from data_processing.preprocess_types import Processing

ENV_FILE = Path(__file__).parents[1] / ".env"


class LightGBMSettings(BaseSettings):
    n_estimators: int = 150
    num_leaves: int = 16
    max_depth: int = 8
    random_state: int = 42
    class_weight: str = "balanced"
    objective: str = "multiclass"


class MlFlowSettings(BaseSettings):
    uri: str = "http://127.0.0.1:5000"
    best_model: str = "models"


class BERTopicParamGrid(BaseSettings):
    min_topic_size: list[int] = [5, 10, 20, 30, 50]
    top_n_words: list[int] = [10, 15, 20]
    n_gram_range: list[tuple[int, int]] = [(1, 1)]


class FigSizes(BaseSettings):
    width: int = 1000
    height: int = 600


class DatasetColumnNames(BaseSettings):
    content: str = "content"
    resume: str = "title"


class TrainingSettings(BaseSettings):
    mlflow: MlFlowSettings = Field(default_factory=MlFlowSettings)
    lightgbm: LightGBMSettings = Field(default_factory=LightGBMSettings)
    figures: FigSizes = Field(default_factory=FigSizes)
    preprocessing: list[Processing] = Field(default=[Processing.RAW])
    columns: DatasetColumnNames = Field(default_factory=DatasetColumnNames)
    params: BERTopicParamGrid = Field(default_factory=BERTopicParamGrid)

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_nested_delimiter="__",
        extra="ignore",
    )


SETTINGS = TrainingSettings()
