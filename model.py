from typing import Callable, Dict, Optional, Sequence, Union
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
import torchmetrics
import ignite
import numpy as np
import src.schnetpack as spk
import src.schnetpack.nn as snn
from src.schnetpack import properties

class RMSE(torchmetrics.Metric):

    def __init__(self):
        self.add_state("sum_squared_errors", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_observations", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.sum_squared_errors += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_errors / self.n_observations)

class MoleculeWise(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_atoms: int = None,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
    ):

        super(MoleculeWise, self).__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        self.aggregation_mode = aggregation_mode
        self.n_in = n_in
        self.n_atoms = n_atoms
        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        if self.aggregation_mode == 'concatenate':
            n_in = n_in * n_atoms

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        representation = inputs["scalar_representation"]

        if self.aggregation_mode == 'concatenate':
            idx_m = inputs[properties.idx_m]
            new_tensor = torch.zeros((int(representation.shape[0] / self.n_atoms), int(representation.shape[1] * self.n_atoms)), 
                                    dtype=representation.dtype, 
                                    device=representation.device)
            for i, idx in enumerate(idx_m):
                j = i % self.n_atoms
                new_tensor[idx, j * self.n_in:(j+1) * self.n_in] = representation[i]
            
            y = self.outnet(new_tensor)

        if self.aggregation_mode == 'sum':
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            representation = snn.scatter_add(representation, idx_m, dim_size=maxm)
            representation = torch.squeeze(representation, -1)

            y = self.outnet(representation)

        if self.aggregation_mode == "avg":
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            representation = snn.scatter_add(representation, idx_m, dim_size=maxm)
            representation = torch.squeeze(representation, -1)
            representation = representation / self.n_atoms

            y = self.outnet(representation)

        inputs[self.output_key] = y
        return inputs

def root_mean_squared_error(pred, targets):
    return torch.linalg.norm(pred - targets)

def weighted_mse(pred, targets, weights):
    targets = targets.reshape(-1, 36**2)
    pred = pred.reshape(-1, 36**2)

    loss = 0
    for i in range(pred.shape[0]):
        loss += torch.sum(torch.square(targets[i] - pred[i]) * weights) 

    return loss / (pred.shape[0] * 36 ** 2)


def get_model(cutoff, n_coeffs, lr):
    # defining the NN
    pairwise_distance = spk.atomistic.PairwiseDistances()
    representation = spk.representation.SchNet(
        n_atom_basis=64, # 256
        n_interactions=5, # 6
        radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )
    # representation = spk.representation.PaiNN(
    #     n_atom_basis=64,
    #     n_interactions=5,
    #     radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
    #     cutoff_fn=spk.nn.CosineCutoff(cutoff)
    # )
    pred_module = spk.atomistic.Atomwise(
        output_key="orbital_coeffs",
        n_in=representation.n_atom_basis,
        n_out=n_coeffs,
    )
    # pred_module = MoleculeWise(
    #     output_key="orbital_coeffs",
    #     aggregation_mode='concatenate',
    #     n_atoms=12,
    #     n_in=representation.n_atom_basis,
    #     n_out=n_coeffs,
    #     activation=torch.tanh
    # )
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
        # loss_weights=torch.from_numpy(np.load('weights.npy')),
        metrics={
            "mse": torchmetrics.regression.MeanSquaredError(),
        },
    )

    # Putting it in the Atomistic Task framework
    task = spk.AtomisticTask(
        model=nnp,
        outputs=[output],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": lr},
        # scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        # scheduler_args={"mode": "min", "patience": 2},
        # scheduler_monitor="val_loss"
    )

    return task