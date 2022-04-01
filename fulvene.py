import logging
import os

import pytorch_lightning
import torch.optim
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger

from src.schnetpack.data.parser import parse_molcas_rasscf_calculation

import src.schnetpack as spk

""" generate ASE database """
# calc_dir = 'C:/users/rhjva/imperial/molcas_files/fulvene/'
# db_path = './test.db'

# for _ in range(29):
#   parse_molcas_rasscf_calculation(calc_dir, db_path)

# from ase.db import connect
# with connect('./test.db') as conn:
#   print(conn.count(), conn.metadata)
  # conn.metadata = {"_distance_unit": 'nm',
                  #  "_property_unit_dict": {"orbital_coeffs": 1.0}}

batch_size = 2
cutoff = 5.0
n_coeffs = 1296

""" Initializing a dataset """
dataset = spk.data.AtomsDataModule(
  datapath='./test.db',
  batch_size=1,
  num_train=0.7,
  num_val=0.3,
  property_units={'orbital_coeffs': 1.0},
  num_workers=8,
  pin_memory=False,
  load_properties=['orbital_coeffs']
)


# defining the NN
representation = spk.representation.SchNet(
    n_atom_basis=64,
    n_interactions=3,
    radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
    cutoff_fn=spk.nn.CosineCutoff(cutoff),
)
pred_module = spk.atomistic.Atomwise(
    output_key="orbital_coeffs",
    n_in=representation.n_atom_basis,
    n_out=n_coeffs
)
nnp = spk.model.NeuralNetworkPotential(
  representation=representation,
  input_modules=None,
  output_modules=[pred_module],
)

# the model output
output = spk.ModelOutput(
    name="orbtital_coeffs",
    loss_fn=torchmetrics.regression.MeanSquaredError(),
    loss_weight=1.0,
    metrics={
        "mse": torchmetrics.regression.MeanSquaredError(),
        "mae": torchmetrics.regression.MeanAbsoluteError(),
    },
)

# Putting it in the Atomistic Task framework
task = spk.AtomisticTask(
    model=nnp,
    outputs=[output],
    optimizer_cls=torch.optim.Adam,
    optimizer_args={"lr": 5e-4},
)

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
        inference_path="best_inference_model"
    ),
    pytorch_lightning.callbacks.EarlyStopping(
        monitor="val_loss", patience=150, mode="min", min_delta=0.0
    ),
    pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
]

logger = TensorBoardLogger("tensorboard/")
trainer = pytorch_lightning.Trainer(callbacks=callbacks, 
                                    logger=logger,
                                    default_root_dir='./test/',
                                    max_epochs=3)

logging.info("Start training")
trainer.fit(task, datamodule=dataset)