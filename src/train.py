import warnings

import lightning as ptl
from lightning.pytorch.callbacks import LearningRateFinder

from lightning_prj.Data.Dataset.dataset import FoodDataModule
from lightning_prj.Model.model import Model
from lightning_prj.Utils.progress_bar import LitProgressBar

warnings.filterwarnings("ignore")

food_data_module = FoodDataModule()
model = Model()

trainer = ptl.Trainer(
    # fast_dev_run=True,
    max_epochs=10,
    accelerator="auto",
    callbacks=[
        LearningRateFinder(),
        LitProgressBar(),
    ],
)

trainer.fit(
    model,
    datamodule=food_data_module,
)
