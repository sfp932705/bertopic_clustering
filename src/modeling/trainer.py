import itertools
import multiprocessing
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from bertopic import BERTopic
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from data_processing.dataset import TsvDataset
from data_processing.preprocess import Processing, prepare_dataset
from modeling.eval import get_coherence_score
from settings import SETTINGS


@dataclass
class BertTrainerConfig:
    documents: list[str]
    params: dict
    processing_type: str


class Trainer:
    def __init__(self, dataset_path: str, experiment_name: str | None = None):
        self.dataset_path = dataset_path
        self.param_grid = SETTINGS.params.dict()
        self.experiment_name = experiment_name if experiment_name else str(uuid.uuid4())
        mlflow.set_tracking_uri(SETTINGS.mlflow.uri)
        mlflow.set_experiment(self.experiment_name)

    def get_dataset(self, processing: Processing):
        dataset: TsvDataset = prepare_dataset(self.dataset_path, processing)
        contents = dataset.get_col_as_numpy(SETTINGS.columns.content)
        return contents

    def get_hyperparams_list(self) -> list[dict]:
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    @staticmethod
    def log_visualizations(model: BERTopic, config: BertTrainerConfig, run_id: str):
        with tempfile.TemporaryDirectory() as _tmp_dir:
            tmp_dir = Path(_tmp_dir)
            try:
                fig = model.visualize_topics(
                    width=SETTINGS.figures.width, height=SETTINGS.figures.height
                )
                fig.write_html(tmp_dir / "vis_topics.html")
            except (ValueError, TypeError):
                pass
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
        with tempfile.TemporaryDirectory() as _tmp_dir:
            tmp_dir = Path(_tmp_dir)
            with (tmp_dir / "topic_info.csv").open("w") as f:
                f.write(model.get_topic_info().to_csv())
            with (tmp_dir / "document_info.csv").open("w") as f:
                f.write(model.get_document_info(config.documents).to_csv())
            mlflow.log_artifacts(
                tmp_dir.as_posix(), artifact_path="info", run_id=run_id
            )

    @staticmethod
    def overfit_small_classifier(features: np.ndarray, target: list[int], run_id: str):
        model = LGBMClassifier(**SETTINGS.lightgbm.model_dump())
        model.fit(features, target)
        predicted = model.predict(features)
        mlflow.log_metric(
            "precision",
            precision_score(target, predicted, average="weighted"),
            run_id=run_id,
        )
        mlflow.log_metric(
            "recall", recall_score(target, predicted, average="weighted"), run_id=run_id
        )
        mlflow.log_metric(
            "f1_score", f1_score(target, predicted, average="weighted"), run_id=run_id
        )

    def fit(self, config: BertTrainerConfig):
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            mlflow.log_params(config.params)
            mlflow.log_param("dataset", self.dataset_path)
            mlflow.log_param("processing", config.processing_type)
            model = BERTopic(**config.params)
            topics, probs = model.fit_transform(config.documents)
            run = mlflow.active_run()
            mlflow.log_metric("clusters", len(model.get_topics()))
            embeddings = model.embedding_model.embed(config.documents)
            self.overfit_small_classifier(embeddings, topics, run.info.run_id)
            self.log_model(model, run.info.run_id)
            self.log_info(model, config, run.info.run_id)
            self.log_visualizations(model, config, run.info.run_id)
            coherence_score = get_coherence_score(model, config.documents, len(topics))
            mlflow.log_metric("coherence_score", coherence_score)

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

    def save_best_model(self):
        output = Path(SETTINGS.mlflow.best_model)
        output.mkdir(exist_ok=True, parents=True)
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        run = mlflow.search_runs(
            experiment.experiment_id, order_by=["metrics.f1_score.DESC"]
        ).iloc[0]
        artifact_uri = run["artifact_uri"]
        mlflow.artifacts.download_artifacts(artifact_uri, dst_path=output)
        processing = run["params.processing"]
        (output / "processing.txt").write_text(processing)
