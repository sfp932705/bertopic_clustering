import polars as pl

from data_processing.dataset import TsvDataset
from data_processing.preprocess_types import Processing
from modeling.fill import Filler
from settings import SETTINGS

CONTENT = SETTINGS.columns.content
SPLIT = "split"
MASKED = "masked"
RESUME = SETTINGS.columns.resume


def explode(dataset: TsvDataset) -> TsvDataset:
    dataset.make_copy_as_list_of_strings(f"{CONTENT}")
    dataset.drop_column(CONTENT)
    dataset.explode(f"{CONTENT}_{SPLIT}")
    return dataset


def fill_masked(dataset: TsvDataset) -> TsvDataset:
    dataset.drop_column(f"{CONTENT}_{SPLIT}")
    dataset.mark_as_masked(f"{CONTENT}_{SPLIT}_{MASKED}")

    masked_dataset = TsvDataset("", load=False)
    masked_dataset.data = dataset.get_masked()
    model = Filler()
    content = list(masked_dataset.get_col_as_numpy(f"{CONTENT}_{SPLIT}_{MASKED}"))
    pred = model.fill_all(content)
    masked_dataset.data = masked_dataset.data.with_columns(
        pl.Series(name=f"{CONTENT}_{SPLIT}_{MASKED}", values=pred)
    )

    dataset.update_with_filled(masked_dataset)
    dataset.data = dataset.data.sort(by=pl.col("Index"))
    dataset.implode(RESUME, f"{CONTENT}_{SPLIT}_{MASKED}")
    dataset.data = dataset.data.rename({f"{CONTENT}_{SPLIT}_{MASKED}": f"{CONTENT}"})
    return dataset


def prepare_dataset(dataset_path: str, processing: Processing):
    dataset = TsvDataset(dataset_path, True)
    dataset.drop_nulls()

    match processing:
        case processing.RAW:
            pass
        case processing.FILL_ATS:
            dataset = explode(dataset)
            dataset.mask_ats(f"{CONTENT}_{SPLIT}")
            dataset = fill_masked(dataset)
        case processing.REMOVE_ATS:
            dataset = explode(dataset)
            dataset.remove_ats(f"{CONTENT}_{SPLIT}")
            dataset.implode(RESUME, f"{CONTENT}_{SPLIT}")
            dataset.data = dataset.data.rename({f"{CONTENT}_{SPLIT}": f"{CONTENT}"})

    return dataset
