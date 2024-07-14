from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from data_processing.preprocess_types import Processing

ENV_FILE = Path(__file__).parents[1] / ".env"


class MlFlowSettings(BaseSettings):
    uri: str = "http://127.0.0.1:5000"
    best_model: str = "models"


class BERTopicParamGrid(BaseSettings):
    min_topic_size: list[int] = [5, 10, 20]
    top_n_words: list[int] = [5, 10]
    n_gram_range: list[tuple[int, int]] = [(1, 1), (1, 2), (1, 3)]


class FigSizes(BaseSettings):
    width: int = 1000
    height: int = 600


class DatasetColumnNames(BaseSettings):
    content: str = "content"
    resume: str = "title"


class TrainingSettings(BaseSettings):
    mlflow: MlFlowSettings = Field(default_factory=MlFlowSettings)
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
