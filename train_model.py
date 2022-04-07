import logging

import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger

import src.schnetpack as spk
import schnetpack.transform as trn
from model import get_model

model_name = 'fulvene_wigner_dist_200'

cutoff = 5.0
n_coeffs = 1296

epochs = 100
batch_size = 16
lr = 5e-4

""" Initializing a dataset """
dataset = spk.data.AtomsDataModule(
  datapath='./data/fulvene_wignerdist_200.db',
  batch_size=batch_size,
  num_train=0.8,
  num_val=0.2,
  transforms=[
    trn.ASENeighborList(cutoff=5.),
    trn.CastTo32()
  ],
  property_units={'orbital_coeffs': 1.0},
  num_workers=0,
  pin_memory=True,
  load_properties=['orbital_coeffs']
)

model = get_model(cutoff, n_coeffs, lr)

# callbacks for PyTroch Lightning Trainer
logging.info("Setup trainer")
callbacks = [
    spk.train.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath="checkpoints",
        filename="{epoch:02d}",
        inference_path="./checkpoints/" + model_name + ".pt"
    ),
    pytorch_lightning.callbacks.EarlyStopping(
        monitor="val_loss", patience=150, mode="min", min_delta=0.0
    ),
    pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
]

logger = WandbLogger(project="schnet-sa-orbitals")
trainer = pytorch_lightning.Trainer(callbacks=callbacks, 
                                    logger=logger,
                                    default_root_dir='./test/',
                                    max_epochs=epochs)

logging.info("Start training")
trainer.fit(model, datamodule=dataset)