import logging
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
import src.schnetpack as spk
import schnetpack.transform as trn

from models.loss_functions import mean_squared_error, rotated_mse, mo_energy_loss, hamiltonian_mse, hamiltonian_mse_energies
from models.orbital_model import get_orbital_model

EXPERIMENTS = [
  'ML_MO',
  "ML_U",
  "ML_F"
]

CUTOFF = 5.0
WAND_PROJECT = 'pyscf-runs'


def train_model(
  property,
  loss_type,
  loss_fn,
  batch_size,
  lr,
  split_file,
  
):
  """ Initializing a dataset """
  dataset = spk.data.AtomsDataModule(
    datapath=database,
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
          inference_path="./checkpoints/" + model_name + ".pt"
      ),
      pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
  ]

  logger = WandbLogger(project=WAND_PROJECT)
  trainer = pytorch_lightning.Trainer(callbacks=callbacks, 
                                      logger=logger,
                                      default_root_dir='./test/',
                                      max_epochs=epochs,
                                      accelerator='gpu',
                                      devices=1)

  logging.info("Start training")
  trainer.fit(model, datamodule=dataset)

def setup_training(experiment_name, batch_size, lr, split_file):
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
    
  train_model(property, loss_type, loss_fn, batch_size, lr, split_file)
    
  
if __name__ == "__main__":  
  model_name = 'geom_scan_200_6-31Gstart_ML_U'
  database = './data/geom_scan_200_6-31G*.db'
  split_file = './data/geom_scan_200.npz'
  epochs = 100
  basis_set_size = 96
  
  experiment_name = 'ML_U'
  
  setup_training(experiment_name, batch_size=16, lr=5e-4, split_file=split_file)