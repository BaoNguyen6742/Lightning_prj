from dataclasses import dataclass


@dataclass
class Config:
    # Dataset configuration
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.1
    TEST_SPLIT: float = 0.1
    NUM_CLASSES: int = 20

    # DataLoader configuration
    BATCH_SIZE: int = 2
    NUM_WORKERS: int = 0

    # Training configuration
    EPOCHS: int = 1000
    LR: float = 0.001


HYPER_PARAMS = Config()
