# import logging
# from typing import Callable
# from models.delta_model import get_delta_model
# from models.loss_functions import hamiltonian_mse
# from models.orbital_model import get_orbital_model
# import pytorch_lightning
# from pytorch_lightning.loggers import WandbLogger
# import torch
# import schnetpack as schnetpack
# from src.schnetpack.data.datamodule import AtomsDataModule

# database_path = './data/geom_scan_199_molcas_ANO-S-MB.db'
# split_file = './data/geom_scan_199_molcas.npz'
# batch_size = 2
# property = 'F'

# dataset = AtomsDataModule(
#   datapath=database_path,
#   batch_size=batch_size,
#   split_file=split_file,
#   transforms=[
#     schnetpack.transform.ASENeighborList(cutoff=5.),
#     schnetpack.transform.CastTo32()
#   ],
#   property_units={property: 1.0, 'guess': 1.0, 'overlap': 1.0},
#   num_workers=0,
#   pin_memory=True,
#   load_properties=[property, 'guess', 'overlap'],
#   is_delta=True
# )

# dataset.setup()

# model = get_delta_model(loss_fn=hamiltonian_mse, loss_type='', lr=1e-4, output_key='F')
# train_dataloader = dataset.train_dataloader()
# for idx, sample in enumerate(dataset.train_dataloader()):
#   output = model(sample)
#   print(output)

""" Let's try inference here """
from db.save_molcas_calculations import save_molcas_calculations_to_db
from openmolcas.utils import read_log_file, write_coeffs_to_orb_file, read_in_orb_file
from models.training import train_model
from models.inference import predict_guess_F, predict_guess_F_delta
import h5py
import numpy as np

prefix = '/home/ubuntu/'
geometry_base_dir = prefix + 'fulvene/geometries/MD_trajectories_5_01/'
calculations_base_dir = prefix + 'fulvene/openmolcas_calculations/MD_trajectory_1/'
working_dir = prefix + 'fulvene/openmolcas_calculations/MD_prev_geometry/'

n_basis = 36
prev_geometry_path = geometry_base_dir + 'geometry_' + str(1) + '.xyz'
curr_geometry_path = geometry_base_dir + 'geometry_' + str(2) + '.xyz'
model_path = './checkpoints/md01_delta_test.pt'

prev_orbs, prev_ener = read_in_orb_file(working_dir + 'geometry_' + str(1) + '/CASSCF.RasOrb')
prev_S = h5py.File(working_dir + 'geometry_' + str(1) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_basis)
prev_F = np.matmul(prev_S, np.matmul(prev_orbs.T, np.matmul(np.diag(prev_ener), np.linalg.inv(prev_orbs.T))))
new_F = predict_guess_F_delta(model_path=model_path, 
                            prev_geometry_path=prev_geometry_path,
                            curr_geometry_path=curr_geometry_path, 
                            F=prev_F,
                            basis=n_basis)

S = h5py.File(working_dir + 'geometry_' + str(2) + '/CASSCF.rasscf.h5').get('AO_OVERLAP_MATRIX')[:].reshape(-1, n_basis)
F = new_F

e_s, U = np.linalg.eig(S)
diag_s = np.diag(e_s ** -0.5)
X = np.dot(U, np.dot(diag_s, U.T))

F_prime = np.dot(X.T, np.dot(F, X))
evals_prime, C_prime = np.linalg.eig(F_prime)
indices = evals_prime.argsort()
C_prime = C_prime[:, indices]
C = np.dot(X, C_prime).T


orbs, evals = read_in_orb_file(working_dir + 'geometry_' + str(2) + '/CASSCF.RasOrb')
print(orbs - C)