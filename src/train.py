import warnings

import lightning as ptl
from lightning.pytorch.callbacks import LearningRateFinder, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

from lightning_prj.Config.config import HYPER_PARAMS
from lightning_prj.Data.Dataset.dataset import FoodDataModule
from lightning_prj.Model.model import Model
from lightning_prj.Utils.progress_bar import LitProgressBar

warnings.filterwarnings("ignore")

food_data_module = FoodDataModule()
model = Model()

mlflow_logger = MLFlowLogger(
    experiment_name="food_classification",
    tracking_uri="file:./mlruns",
)
trainer = ptl.Trainer(
    # fast_dev_run=True,
    max_epochs=HYPER_PARAMS.EPOCHS,
    accelerator="auto",
    callbacks=[
        LearningRateFinder(),
        LitProgressBar(),
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=True,
            log_weight_decay=True,
        ),
    ],
    logger=mlflow_logger,
    accumulate_grad_batches=64,
    log_every_n_steps=1,
)

trainer.fit(
    model,
    datamodule=food_data_module,
)
