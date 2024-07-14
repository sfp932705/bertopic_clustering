from argparse import ArgumentParser
from pathlib import Path

from modeling.trainer import Trainer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    trainer = Trainer(dataset_path=args.dataset.absolute().as_posix())
    trainer.train()
