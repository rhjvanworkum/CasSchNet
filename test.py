import logging
from typing import Callable
from models.delta_model import get_delta_model
from models.loss_functions import hamiltonian_mse
from models.orbital_model import get_orbital_model
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
import torch
import schnetpack as schnetpack
from src.schnetpack.data.datamodule import AtomsDataModule

database_path = './data/geom_scan_199_molcas_ANO-S-MB.db'
split_file = './data/geom_scan_199_molcas.npz'
batch_size = 2
property = 'F'

dataset = AtomsDataModule(
  datapath=database_path,
  batch_size=batch_size,
  split_file=split_file,
  transforms=[
    schnetpack.transform.ASENeighborList(cutoff=5.),
    schnetpack.transform.CastTo32()
  ],
  property_units={property: 1.0, 'guess': 1.0, 'overlap': 1.0},
  num_workers=0,
  pin_memory=True,
  load_properties=[property, 'guess', 'overlap'],
  is_delta=True
)

dataset.setup()

model = get_delta_model(loss_fn=hamiltonian_mse, loss_type='', lr=1e-4, output_key='F')

# print(dataset.train_dataset[0]['mo_coeffs_adjusted'].shape)

train_dataloader = dataset.train_dataloader()
for idx, sample in enumerate(dataset.train_dataloader()):
  output = model(sample)
  print(output)

# print(F.shape)

# from ase.db import connect
# import numpy as np
# with connect(database_path) as conn:
#   # print(conn.get(1).data['F'].shape)
#   F_2 = conn.get(2).data['F'].reshape(36, 36)
#   assert np.allclose(F, F_2)