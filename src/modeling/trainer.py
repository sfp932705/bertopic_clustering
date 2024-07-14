import itertools
import multiprocessing
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
from bertopic import BERTopic

from data_processing.dataset import TsvDataset
from data_processing.preprocess import CONTENT, Processing, prepare_dataset
from settings import SETTINGS

PROCESSINGS = [Processing.RAW]


@dataclass
class BertTrainerConfig:
    documents: list[str]
    params: dict
    processing_type: str


class Trainer:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.param_grid = SETTINGS.params.dict()

    def get_dataset(self, processing: Processing):
        dataset: TsvDataset = prepare_dataset(self.dataset_path, processing)
        contents = dataset.get_col_as_numpy(CONTENT)
        return contents

    def get_hyperparams_list(self) -> list[dict]:
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    @staticmethod
    def log_visualizations(model: BERTopic, config: BertTrainerConfig, run_id: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            fig = model.visualize_topics(
                width=SETTINGS.figures.width, height=SETTINGS.figures.height
            )
            fig.write_html(tmp_dir / "vis_topics.html")
            fig = model.visualize_hierarchy(
                width=SETTINGS.figures.width, height=SETTINGS.figures.height
            )
            fig.write_html(tmp_dir / "vis_hierarchy.html")
            fig = model.visualize_documents(
                config.documents,
                width=SETTINGS.figures.width,
                height=SETTINGS.figures.height,
            )
            fig.write_html(tmp_dir / "vis_documents.html")
            mlflow.log_artifacts(
                tmp_dir.as_posix(), artifact_path="visualizations", run_id=run_id
            )

    @staticmethod
    def log_model(model: BERTopic, run_id: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir, serialization="safetensors")
            mlflow.log_artifacts(tmp_dir, artifact_path="model", run_id=run_id)

    @staticmethod
    def log_info(model: BERTopic, config: BertTrainerConfig, run_id: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            with (tmp_dir / "topic_info.csv").open("w") as f:
                f.write(model.get_topic_info().to_csv())
            with (tmp_dir / "document_info.csv").open("w") as f:
                f.write(model.get_document_info(config.documents).to_csv())
            mlflow.log_artifacts(
                tmp_dir.as_posix(), artifact_path="info", run_id=run_id
            )

    def fit(self, config: BertTrainerConfig):
        mlflow.set_tracking_uri(SETTINGS.mlflow.uri)
        with mlflow.start_run():
            mlflow.log_params(config.params)
            mlflow.log_param("processing", config.processing_type)
            model = BERTopic(**config.params)
            model.fit_transform(config.documents)
            run = mlflow.active_run()
            self.log_model(model, run.info.run_id)
            self.log_info(model, config, run.info.run_id)
            self.log_visualizations(model, config, run.info.run_id)

    def train(self):
        pool = multiprocessing.Pool(processes=4)
        configs = []
        for processing in SETTINGS.preprocessing:
            documents = self.get_dataset(processing).tolist()
            for params in self.get_hyperparams_list():
                config = BertTrainerConfig(
                    documents=documents, params=params, processing_type=processing.value
                )
                configs.append(config)
        pool.map(self.fit, configs)
