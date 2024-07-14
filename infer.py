from argparse import ArgumentParser
from pathlib import Path

from modeling.infer import Infer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="inferences")
    args = parser.parse_args()
    predictor = Infer(
        dataset_path=args.dataset.absolute(),
        model_path=args.model.absolute(),
        output_path=args.output.absolute(),
    )
    predictor.infer()
