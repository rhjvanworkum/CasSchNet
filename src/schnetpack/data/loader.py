import torch
from torch.utils.data import DataLoader

from typing import Optional, Sequence
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, T_co

import schnetpack.properties as structure

__all__ = ["AtomsLoader"]


def _atoms_collate_fn(batch):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    elem = batch[0]
    idx_keys = {structure.idx_i, structure.idx_j, structure.idx_i_triples}
    # Atom triple indices must be treated separately
    idx_triple_keys = {structure.idx_j_triples, structure.idx_k_triples}

    coll_batch = {}
    for key in elem:
        if (key not in idx_keys) and (key not in idx_triple_keys):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch[structure.n_atoms], dim=0)
    seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch)), repeats=coll_batch[structure.n_atoms], dim=0
    )
    coll_batch[structure.idx_m] = idx_m

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in elem.keys():
            indices = []
            offset = 0
            for idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d[structure.idx_j].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch


def _atoms_collate_double_fn(batch):
    elem = batch[0]
    idx_keys = {structure.idx_i, structure.idx_j, structure.idx_i_triples}
    # Atom triple indices must be treated separately
    idx_triple_keys = {structure.idx_j_triples, structure.idx_k_triples}
    # property_keys
    property_keys = {'F', 'mo_coeffs_adjusted'}

    coll_batch = {}
    for key in elem:
        if key in property_keys:
            coll_batch[key] = torch.cat([torch.sub(d[key][1], d[key][0]) for d in batch], 0)
        elif (key not in idx_keys) and (key not in idx_triple_keys):
            coll_batch[key] = torch.cat([d[key] for d in batch], 1)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 1)


    idx_m_list = []
    idx_i = []
    idx_j = []
    idx_i_triples = []
    for idx in range(2):
        seg_m = torch.cumsum(coll_batch[structure.n_atoms][idx], dim=0)
        seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
        idx_m = torch.repeat_interleave(
            torch.arange(len(batch)), repeats=coll_batch[structure.n_atoms][idx], dim=0
        )
        idx_m_list.append(idx_m)

        for key, list in zip(idx_keys, [idx_i, idx_j, idx_i_triples]):
            if key in elem.keys():
                list.append(torch.cat(
                    [d[key][idx] + off for d, off in zip(batch, seg_m)], 0
                ))

    coll_batch[structure.idx_m] = torch.stack(idx_m_list)
    for key, list in zip(idx_keys, [idx_i, idx_j, idx_i_triples]):
        if len(list) != 0:
            coll_batch[key] = torch.stack(list)
            coll_batch[key] = coll_batch[key].long()

    return coll_batch


class AtomsLoader(DataLoader):
    """Data loader for subclasses of BaseAtomsData"""

    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t = _atoms_collate_fn,
        pin_memory: bool = False,
        is_delta: bool = False,
        **kwargs
    ):

        if is_delta:
            collate_fn = _atoms_collate_double_fn

        super(AtomsLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **kwargs
        )
