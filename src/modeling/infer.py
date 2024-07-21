from pathlib import Path

from bertopic import BERTopic

from data_processing.dataset import TsvDataset
from data_processing.preprocess import Processing, prepare_dataset
from settings import SETTINGS


class Infer:
    def __init__(self, dataset_path: Path, model_path: Path, output_path: Path):
        self.model = BERTopic.load((model_path / "artifacts" / "model").as_posix())
        self.processing = self.get_processing(model_path / "processing.txt")
        self.dataset_path = dataset_path.as_posix()
        self.output_path = Path(output_path)

    def get_processing(self, processing_txt: Path) -> Processing:
        return Processing(processing_txt.read_text())

    def get_dataset(self, processing: Processing):
        dataset: TsvDataset = prepare_dataset(self.dataset_path, processing)
        contents = dataset.get_col_as_numpy(SETTINGS.columns.content)
        return contents

    def save_visualizations(self, documents: list[str]):
        try:
            fig = self.model.visualize_topics(
                width=SETTINGS.figures.width, height=SETTINGS.figures.height
            )
            fig.write_html(self.output_path / "vis_topics.html")
        except (ValueError, TypeError):
            pass
        fig = self.model.visualize_hierarchy(
            width=SETTINGS.figures.width, height=SETTINGS.figures.height
        )
        fig.write_html(self.output_path / "vis_hierarchy.html")
        fig = self.model.visualize_documents(
            documents,
            width=SETTINGS.figures.width,
            height=SETTINGS.figures.height,
        )
        fig.write_html(self.output_path / "vis_documents.html")

    def log_preds(self, topics: list[int]):
        with (self.output_path / "preds.txt").open("w") as f:
            f.write("\n".join(str(x) for x in topics))

    def log_info(self, documents: list[str]):
        with (self.output_path / "topic_info.csv").open("w") as f:
            f.write(self.model.get_topic_info().to_csv())
        with (self.output_path / "document_info.csv").open("w") as f:
            f.write(self.model.get_document_info(documents).to_csv())

    def infer(self):
        documents = self.get_dataset(self.processing).tolist()
        topics, probs = self.model.transform(documents)
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.log_preds(topics.tolist())
        self.log_info(documents)
        self.save_visualizations(documents)
