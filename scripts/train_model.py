from models.training import train_model
from models.loss_functions import mean_squared_error, rotated_mse, mo_energy_loss, hamiltonian_mse, hamiltonian_mse_energies

EXPERIMENTS = [
  'ML_MO',
  "ML_U",
  "ML_F"
]
CUTOFF = 5.0
WAND_PROJECT = 'new'

if __name__ == "__main__":  
  model_name = 'md01_delta_test'
  database_path = './data/MD01_molcas_ANO-S-MB.db'
  # database_path = './data/geom_scan_199_molcas_ANO-S-MB.db'
  split_file = './data/geom_scan_200_molcas.npz'
  epochs = 100
  lr=5e-4
  batch_size=16
  basis_set_size = 36
  experiment_name = 'ML_F'
  is_delta = True
  use_wandb = True

  if is_delta:
    cutoff = 7.0
  else:
    cutoff = 5.0

  if experiment_name not in EXPERIMENTS:
    raise ValueError("invalid experiment name")
  
  if experiment_name == 'ML_MO':
    property = 'mo_coeffs_adjusted'
    loss_type = ''
    loss_fn = mean_squared_error
  elif experiment_name == 'ML_U':
    property = 'mo_coeffs'
    loss_type = 'reference'
    loss_fn = rotated_mse
  elif experiment_name == 'ML_F':
    property = 'F'
    loss_type = ''
    loss_fn = hamiltonian_mse
    
  train_model(save_path='./checkpoints/' + model_name + '.pt',
                  property=property, 
                  loss_type=loss_type, 
                  loss_fn=loss_fn, 
                  batch_size=batch_size, 
                  lr=lr, 
                  epochs=epochs,
                  basis_set_size=basis_set_size,
                  database_path=database_path,
                  split_file=split_file,
                  use_wandb=use_wandb,
                  is_delta=is_delta,
                  cutoff=cutoff) 