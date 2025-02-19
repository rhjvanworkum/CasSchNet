import torch
import torch.optim
import torchmetrics
import src.schnetpack as spk
from src.schnetpack.task import ModelOutput
from models.utils import Fake

from models.delta_neural_network_potential import DeltaNeuralNetworkPotential

def get_delta_model(loss_fn, loss_type, lr, output_key, basis_set_size=36, cutoff=5.0):
    pairwise_distance = spk.atomistic.PairwiseDistances()
    representation = spk.representation.PaiNN(
        n_atom_basis=32,
        n_interactions=5,
        radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
    pred_module = spk.atomistic.Atomwise(
        output_key=output_key,
        n_in=representation.n_atom_basis * 2,
        n_layers=2,
        n_out=basis_set_size**2
    )
    nnp = DeltaNeuralNetworkPotential(
        representation=representation,
        input_modules=[pairwise_distance],
        output_modules=[pred_module],
    )

    output = ModelOutput(
        name=output_key,
        loss_fn=loss_fn,
        loss_weight=1.0,
        loss_type=loss_type,
        basis_set_size=basis_set_size
    )

    # Putting it in the Atomistic Task framework
    task = spk.AtomisticTask(
        model=nnp,
        outputs=[output, Fake('overlap'), Fake('guess')],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": lr},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={'threshold': 1e-6, 'patience': 10},
        scheduler_monitor='val_loss'
    )

    return task
