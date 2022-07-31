from __future__ import annotations
from mimetypes import init

from typing import Dict, Optional, List

from schnetpack.transform import Transform
import schnetpack.properties as properties
from src.schnetpack.model.base import AtomisticModel

import torch
import torch.nn as nn

class DeltaNeuralNetworkPotential(AtomisticModel):
    """
    A generic neural network potential class that sequentially applies a list of input modules, a representation module
    and a list of output modules.

    This can be flexibly configured for various, e.g. property prediction or potential energy sufaces with response
    properties.
    """

    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        postprocessors: Optional[List[Transform]] = None,
        input_dtype: torch.dtype = torch.float32,
        do_postprocessing: Optional[bool] = None,
    ):
        """
        Args:
            representation: The module that builds representation from inputs.
            input_modules: Modules that are applied before representation, e.g. to modify input or add additional tensors for response
                properties.
            output_modules: Modules that predict output properties from the representation.
            postprocessors: Post-processing transforms tha may be initialized using te `datamodule`, but are not
                applied during training.
            input_dtype: The dtype of real inputs.
            do_postprocessing: If true, post-processing is activated.
        """
        super().__init__(
            input_dtype=input_dtype,
            postprocessors=postprocessors,
            do_postprocessing=do_postprocessing,
        )
        self.representation = representation
        self.input_modules = nn.ModuleList(input_modules)
        self.output_modules = nn.ModuleList(output_modules)

        self.collect_derivatives()
        self.collect_outputs()

    def _split_and_collect(self, inputs, callable):
        outputs = []

        for i in range(2):
            inputs_split = {}
            for key, value in inputs.items():
                inputs_split[key] = value[i]
            outputs.append(callable(inputs_split))

        inputs = {}
        for key in outputs[0].keys():
            inputs[key] = torch.stack((outputs[0][key], outputs[1][key]))

        return inputs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:        
        # inititalize derivatives for response properties
        initial_inputs = self.initialize_derivatives(inputs)

        for m in self.input_modules:
            inputs = self._split_and_collect(initial_inputs, m)

        inputs = self._split_and_collect(inputs, self.representation)

        inputs['_idx_m'] = inputs['_idx_m'][0]
        n_atoms_in_batch = inputs['_atomic_numbers'].shape[1]
        inputs['scalar_representation'] = inputs['scalar_representation'].reshape(n_atoms_in_batch, -1)

        inputs['F'] = initial_inputs['F']
        inputs['mo_coeffs_adjusted'] = initial_inputs['mo_coeffs_adjusted']

        for m in self.output_modules:
            inputs = m(inputs)

        # apply postprocessing (if enabled)
        inputs = self.postprocess(inputs)
        results = self.extract_outputs(inputs)

        return results
