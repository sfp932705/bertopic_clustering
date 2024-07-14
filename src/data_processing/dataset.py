from __future__ import annotations

import typing
from pathlib import Path
from random import randint

import polars as pl

from data_processing.stop_words import STOP_WORDS


class TsvDataset:
    def __init__(self, dataset_path: str, load: bool = True):
        self.data: pl.DataFrame = pl.DataFrame()
        if load:
            self.load(dataset_path)
            self.add_index()

    @typing.no_type_check
    def check_loaded_data(func):
        def wrapper(self, *args, **kwargs):
            if not len(self.data):
                raise ValueError("LOAD DATA FIRST.")
            return func(self, *args, **kwargs)

        return wrapper

    def load(self, dataset_path: str):
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Unable to find {dataset_path}")
        self.data = pl.read_csv(dataset_path, separator="\t", encoding="utf-8")

    @check_loaded_data
    def add_index(self):
        self.data = self.data.with_row_index("Index")

    @check_loaded_data
    def __len__(self):
        if self.data is not None:
            return len(self.data)
        return 0

    @check_loaded_data
    def drop_nulls(self):
        self.data = self.data.drop_nulls()

    @typing.no_type_check
    @check_loaded_data
    def get_row(self, row: int, named: bool = False):
        return self.data.row(index=row, named=named)

    @check_loaded_data
    def get_col(self, column: str):
        return self.data.select(pl.col(column))

    @check_loaded_data
    def get_col_as_numpy(self, column: str):
        return self.get_col(column).to_numpy()[:, 0]

    @check_loaded_data
    def drop_column(self, column_name: str):
        self.data = self.data.drop(column_name)

    def sample(self):
        return self.get_row(randint(0, len(self)))

    @check_loaded_data
    def make_copy_as_list_of_strings(self, column: str):
        self.data = self.data.with_columns(
            pl.col(column).str.split("\n").alias(f"{column}_split")
        )

    @check_loaded_data
    def explode(self, column: str | list[str]):
        self.data = self.data.explode(column)

    @check_loaded_data
    def implode(self, column_on: str, columns: str | list[str]):
        self.data = self.data.group_by(column_on).agg(pl.col(columns).str.concat(" "))

    def count_words(self, column: str):
        return (
            self.data.select(
                pl.col(column)
                .str.replace_all(r"[^\w\s]", "")  # remove punctuations
                .str.to_lowercase()
                .str.split(" ")
                .list.set_difference(STOP_WORDS)
                .flatten()
                .alias("words")
            )
            .to_series()
            .value_counts()
            .filter(pl.col("words").len() > 0)
            .sort(by="count", descending=True)
        )

    @check_loaded_data
    def get_at_counts(self, column: str):
        return (
            self.data.select(
                pl.col(column)
                .str.replace_all(" ", "")
                .str.replace_all(r"[^\@]", " ")
                .str.split(" ")
                .flatten()
                .alias("words")
            )
            .to_series()
            .value_counts()
            .filter(pl.col("words").len() > 0)
            .sort(by="count", descending=True)
        )

    @check_loaded_data
    def mask_ats(self, column: str):
        self.data = self.data.with_columns(
            pl.col(column)
            .str.replace_all(r"@[\s@]+@", "[MASK]")
            .alias(f"{column}_masked")
        )

    @check_loaded_data
    def remove_ats(self, column: str):
        self.data = self.data.with_columns(
            pl.col(column).str.replace_all(r"@[\s@]+@", "")
        )

    @check_loaded_data
    def mark_as_masked(self, column: str):
        self.data = self.data.with_columns(
            (pl.col(column).str.contains(r"\[MASK\]")).alias("is_masked")
        )

    @check_loaded_data
    def get_masked(self):
        return self.data.filter(pl.col("is_masked"))

    @check_loaded_data
    def get_non_masked(self):
        return self.data.filter(~pl.col("is_masked"))

    @check_loaded_data
    def update_with_filled(self, other: TsvDataset):
        self.data = pl.concat([self.get_non_masked(), other.data])
