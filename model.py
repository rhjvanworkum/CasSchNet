import torch.optim
import torchmetrics
import src.schnetpack as spk

def get_model(cutoff, n_coeffs, lr):
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
        optimizer_args={"lr": lr},
    )

    return task