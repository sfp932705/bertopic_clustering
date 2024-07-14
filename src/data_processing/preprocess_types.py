from enum import Enum


class Processing(Enum):
    RAW = "RAW"
    FILL_ATS = "FILL_ATS"
    REMOVE_ATS = "REMOVE_ATS"
