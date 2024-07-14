import math
from enum import Enum

import torch
from transformers import BertForMaskedLM, BertTokenizer


class Model(Enum):
    UNCASED = "bert-base-uncased"
    CASED = "bert-base-cased"
    LARGE_UNCASED = "bert-large-uncased"
    LARGE_CASED = "bert-large-cased"


class Filler:
    def __init__(self, model: Model = Model.CASED):
        self.tokenizer = BertTokenizer.from_pretrained(model.value)
        self.model = BertForMaskedLM.from_pretrained(model.value)

    def predict(self, input_text):
        encoded_input = self.tokenizer(input_text, return_tensors="pt", padding=True)
        mask_index = self.tokenizer.mask_token_id == encoded_input["input_ids"]
        with torch.no_grad():
            predictions = self.model(**encoded_input)
        pred_tokens = torch.argmax(predictions[0], dim=2)
        encoded_input["input_ids"][mask_index] = pred_tokens[mask_index]
        return self.tokenizer.decode(
            encoded_input["input_ids"].flatten().tolist(), skip_special_tokens=False
        )

    def fill(self, sentence: str) -> str:
        return self.fill_batch([sentence])[0]

    def fill_batch(self, sentence: list[str]) -> list[str]:
        pred = self.predict(sentence)
        filtered = (
            pred.replace(" [CLS]", "")
            .replace("[CLS] ", "")
            .replace("[PAD] ", "")
            .split(" [SEP]")
        )[:-1]
        filtered[-1] = filtered[-1].replace(" [SEP]", "")
        return filtered

    def fill_all(self, data: list, batch_size: int = 64):
        preds = []
        for batch in range(math.ceil(len(data) / batch_size)):
            start = batch_size * batch
            end = start + batch_size
            preds.extend(self.fill_batch(data[start:end]))
        return preds
