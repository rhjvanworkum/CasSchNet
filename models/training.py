import logging
from typing import Callable
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
import torch
import src.schnetpack as spk
import schnetpack.transform as trn

from models.loss_functions import mean_squared_error, rotated_mse, mo_energy_loss, hamiltonian_mse, hamiltonian_mse_energies
from models.orbital_model import get_orbital_model

WAND_PROJECT = 'new'

def train_model(
  save_path: str,
  property: str = 'F',
  loss_type: str = '',
  loss_fn: Callable = hamiltonian_mse,
  batch_size: int = 16,
  lr: float = 5e-4,
  epochs: int = 100,
  basis_set_size: int = 36,
  database_path: str = None,
  split_file: str = None,
  use_wandb: bool = False,
  initial_model_path: str = None
  
):
  """ Initializing a dataset """
  dataset = spk.data.AtomsDataModule(
    datapath=database_path,
    batch_size=batch_size,
    split_file=split_file,
    transforms=[
      trn.ASENeighborList(cutoff=5.),
      trn.CastTo32()
    ],
    property_units={property: 1.0, 'guess': 1.0, 'overlap': 1.0},
    num_workers=0,
    pin_memory=True,
    load_properties=[property, 'guess', 'overlap']
  )

  """ Initiating the Model """
  model = get_orbital_model(loss_fn=loss_fn, loss_type=loss_type, lr=lr, output_key=property, basis_set_size=basis_set_size)

  if initial_model_path is not None:
    model.load_state_dict(torch.load(initial_model_path))

  """ Just for testing purposes """
  # dataset.setup()
  # for idx, sample in enumerate(dataset.train_dataloader()):
  #   # output = model(sample)
  #   loss = model.training_step(sample, 0)
  #   # loss = model.training_step(sample, 1)
  #   print(loss)
  #   break


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
          inference_path=save_path
      ),
      pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
  ]

  if use_wandb:
    logger = WandbLogger(project=WAND_PROJECT)
    trainer = pytorch_lightning.Trainer(callbacks=callbacks, 
                                        logger=logger,
                                        default_root_dir='./test/',
                                        max_epochs=epochs,
                                        # accelerator='gpu',
                                        # deterministic=True,
                                        devices=1)
  else:
    trainer = pytorch_lightning.Trainer(callbacks=callbacks, 
                                    default_root_dir='./test/',
                                    max_epochs=epochs,
                                    # accelerator='gpu',
                                    # deterministic=True,
                                    devices=1)
  logging.info("Start training")
  trainer.fit(model, datamodule=dataset)