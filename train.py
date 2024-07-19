from argparse import ArgumentParser
from pathlib import Path

from modeling.trainer import Trainer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment_name", type=str, default=None)
    args = parser.parse_args()
    trainer = Trainer(
        dataset_path=args.dataset.absolute().as_posix(),
        experiment_name=args.experiment_name,
    )
    trainer.train()
    trainer.save_best_model()
