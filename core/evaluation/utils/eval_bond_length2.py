"""Utils for evaluating bond length."""

import collections
from typing import Tuple, Sequence, Dict, Optional

import numpy as np
from scipy import spatial as sci_spatial
import matplotlib.pyplot as plt

import core.datasets.utils as utils_data

from rdkit import Chem


BondType = Tuple[int, int, int]  # (atomic_num, atomic_num, bond_type)
BondLengthData = Tuple[BondType, float]  # (bond_type, bond_length)
BondLengthProfile = Dict[BondType, np.ndarray]  # bond_type -> empirical distribution

BOND_TYPES = frozenset((
    (6, 6, 1), #	CC	716	29.6%	1.2857153558712793	1.696778883283098	0.004110635274118186
    (6, 6, 4), #	C:C	500	20.7%	1.2981754588686738	1.5429516779717267	0.002447762191030529
    (6, 8, 1), #	CO	336	13.9%	1.217717567891834	1.592581263775381	0.0037486369588354694
    (6, 7, 1), #	CN	245	10.1%	1.2412786652760066	1.609101379383609	0.0036782271410760246
    (6, 7, 4), #	C:N	213	8.8%	1.2781037555594505	1.4881754271876604	0.002100716716282098
))

BOND_LENGTH_BINS = np.arange(1.1, 1.700001, 0.005)


def _bond_str(bond_type: int) -> str:
    assert bond_type in [1, 2, 3, 4]
    if bond_type == 1:
        bond_str = ''
    elif bond_type == 2:
        bond_str = '='
    elif bond_type == 3:
        bond_str = '#'
    elif bond_type == 4:
        bond_str = ':'
    return bond_str


def _atom_str(atomic_num) -> str:
    # convert atomic numbers to atom symbols
    return Chem.GetPeriodicTable().GetElementSymbol(atomic_num)


def _bond_type_str(bond_type: BondType) -> str:
    s_num, s_hybrid, s_aromatic, e_num, e_hybrid, e_aromatic, _type = bond_type
    return f'{_atom_str(s_num)}{s_hybrid}{s_aromatic}{_bond_str(_type)}{_atom_str(e_num)}{e_hybrid}{e_aromatic}'


def _format_bond_type(bond_type: BondType) -> BondType:
    s_num, s_hybrid, s_aromatic, e_num, e_hybrid, e_aromatic, _type = bond_type
    if s_num > e_num:
        s_num, s_hybrid, s_aromatic, e_num, e_hybrid, e_aromatic = e_num, e_hybrid, e_aromatic, s_num, s_hybrid, s_aromatic
    return s_num, s_hybrid, s_aromatic, e_num, e_hybrid, e_aromatic, _type


def bond_distance_from_mol(mol):
    pos = mol.GetConformer().GetPositions()
    pdist = pos[None, :] - pos[:, None]
    pdist = np.sqrt(np.sum(pdist ** 2, axis=-1))
    all_distances = []
    for bond in mol.GetBonds():
        s_atom = bond.GetBeginAtom()
        e_atom = bond.GetEndAtom()

        s_num = s_atom.GetAtomicNum()
        e_num = e_atom.GetAtomicNum()

        # 获取杂化类型
        s_hybrid = s_atom.GetHybridization().name
        e_hybrid = e_atom.GetHybridization().name

        # 获取芳香性
        s_aromatic = s_atom.GetIsAromatic()
        e_aromatic = e_atom.GetIsAromatic()

        s_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        _type = utils_data.BOND_TYPES[bond.GetBondType()]
        distance = pdist[s_idx, e_idx]

        bond_type = (s_num, s_hybrid, s_aromatic, e_num, e_hybrid, e_aromatic, _type)
        bond_type = _format_bond_type(bond_type)


        all_distances.append((bond_type, distance))
    return all_distances


def get_distribution(distances: Sequence[float], bins=BOND_LENGTH_BINS) -> np.ndarray:
    return np.histogram(distances, bins=bins, density=True)[0]


def get_bond_lengths(bond_lengths: Sequence[BondLengthData]) -> BondLengthProfile:
    bond_length_profile = collections.defaultdict(list)
    for bond_type, bond_length in bond_lengths:
        bond_type = _format_bond_type(bond_type)
        bond_length_profile[bond_type].append(bond_length)
    return bond_length_profile


def get_bond_length_profile(bond_lengths: Sequence[BondLengthData]) -> BondLengthProfile:
    bond_length_profile = get_bond_lengths(bond_lengths)
    bond_length_profile = {k: get_distribution(v) 
        for k, v in bond_length_profile.items() if k in BOND_TYPES}
    return bond_length_profile


def eval_bond_length_profile(ref_bond_length_profile: BondLengthProfile, 
                             bond_length_profile: BondLengthProfile) -> Dict[str, Optional[float]]:
    metrics = {}

    # Jensen-Shannon distances
    for bond_type in BOND_TYPES:
        if bond_type in bond_length_profile:
            metrics[f'JSD_{_bond_type_str(bond_type)}'] = sci_spatial.distance.jensenshannon(
                ref_bond_length_profile[bond_type],
                bond_length_profile[bond_type]
            )

    return metrics
