import logging
import os

import pytorch_lightning
import torch.optim
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger

import src.schnetpack as spk
import schnetpack.transform as trn

batch_size = 2
cutoff = 5.0
n_coeffs = 1296

""" Initializing a dataset """
dataset = spk.data.AtomsDataModule(
  datapath='./data/test.db',
  batch_size=batch_size,
  num_train=10,
  num_val=5,
  transforms=[
    trn.ASENeighborList(cutoff=5.),
    trn.CastTo32()
  ],
  property_units={'orbital_coeffs': 1.0},
  num_workers=0,
  pin_memory=True,
  load_properties=['orbital_coeffs']
)

# defining the NN
pairwise_distance = spk.atomistic.PairwiseDistances()
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
  input_modules=[pairwise_distance],
  output_modules=[pred_module],
)

# the model output
output = spk.ModelOutput(
    name="orbital_coeffs",
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
        inference_path="best_inference_model.pt"
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
                                    max_epochs=5)

logging.info("Start training")
trainer.fit(task, datamodule=dataset)