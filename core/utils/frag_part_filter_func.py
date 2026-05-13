import os
import re
import numpy as np
import torch
import torch_scatter
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def get_batch_connectivity_matrix(ligand_batch, ligand_bond_index, ligand_bond_type, ligand_bond_batch):
    batch_ligand_size = torch_scatter.segment_coo(
        torch.ones_like(ligand_batch),
        ligand_batch,
        reduce='sum',
    )
    batch_index_offset = torch.cumsum(batch_ligand_size, 0) - batch_ligand_size
    batch_size = len(batch_index_offset)
    batch_connectivity_matrix = []
    for batch_index in range(batch_size):
        start_index, end_index = ligand_bond_index[:, ligand_bond_batch == batch_index]
        start_index -= batch_index_offset[batch_index]
        end_index -= batch_index_offset[batch_index]
        bond_type = ligand_bond_type[ligand_bond_batch == batch_index]
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.
        connectivity_matrix = torch.zeros(batch_ligand_size[batch_index], batch_ligand_size[batch_index],
                                          dtype=torch.int)
        for s, e, t in zip(start_index, end_index, bond_type):
            connectivity_matrix[s, e] = connectivity_matrix[e, s] = t
        batch_connectivity_matrix.append(connectivity_matrix)
    return batch_connectivity_matrix


def get_batch_type_pmf_matrix(ligand_batch, ligand_bond_index, ligand_bond_type_pmf, ligand_bond_batch, padding=False, num_atoms_max=65):
    batch_ligand_size = torch_scatter.segment_coo(
        torch.ones_like(ligand_batch),
        ligand_batch,
        reduce='sum',
    )
    batch_index_offset = torch.cumsum(batch_ligand_size, 0) - batch_ligand_size
    batch_size = len(batch_index_offset)
    batch_connectivity_matrix = []
    E = ligand_bond_type_pmf.size(-1)
    max_N = num_atoms_max  # Maximum number of atoms in a ligand (magic number for crossdock)
    for batch_index in range(batch_size):
        start_index, end_index = ligand_bond_index[:, ligand_bond_batch == batch_index]
        start_index -= batch_index_offset[batch_index]
        end_index -= batch_index_offset[batch_index]
        bond_type_pmf = ligand_bond_type_pmf[ligand_bond_batch == batch_index]
        # NxNxE connectivity matrix where each bond_type_pmf represents a vector of float densities over bond type (0: none, 1: single, 2: double, 3: triple, 4: aromatic).
        if padding:
            connectivity_matrix = torch.zeros(
               (batch_ligand_size[batch_index], max_N, E), dtype=torch.float 
            )
        else:
            connectivity_matrix = torch.zeros(
                (batch_ligand_size[batch_index], batch_ligand_size[batch_index], E), dtype=torch.float
            )
        for s, e, t in zip(start_index, end_index, bond_type_pmf):
            connectivity_matrix[s, e] = connectivity_matrix[e, s] = t
        batch_connectivity_matrix.append(connectivity_matrix)
    return batch_connectivity_matrix

def bfs_substructure_mask(ligand_pos, batch_ligand, substructure_size, distance_threshold=1.8):
    """
    ligand_pos: (N, 3) tensor of coordinates
    batch_ligand: (N,) tensor of molecule indices
    m: target substructure size
    distance_threshold: threshold to define bonds (in Å)
    """
    N = ligand_pos.size(0)
    mask = torch.zeros(N, dtype=torch.bool)

    unique_mols = batch_ligand.unique()

    for mol_idx in unique_mols:
        mol_atom_idx = torch.nonzero(batch_ligand == mol_idx, as_tuple=False).squeeze(-1)
        if mol_atom_idx.numel() == 0:
            continue

        coords = ligand_pos[mol_atom_idx]
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist = torch.linalg.norm(diff, dim=-1)
        adjacency = (dist < distance_threshold) & (dist > 0)

        num_atoms = len(mol_atom_idx)
        start = torch.randint(0, num_atoms, (1,)).item()

        # ✅ 用布尔 mask 来记录访问状态
        visited_mask = torch.zeros(num_atoms, dtype=torch.bool).to(ligand_pos.device)
        visited_mask[start] = True

        # 用 torch 实现 queue：记录当前层原子索引
        frontier = torch.tensor([start], dtype=torch.long).to(ligand_pos.device)

        while frontier.numel() > 0 and visited_mask.sum() < substructure_size:
            neighbors = adjacency[frontier].any(dim=0)  # 当前层的邻居集合
            new_frontier = neighbors & (~visited_mask)   # 还没访问过的邻居
            visited_mask |= new_frontier                 # 更新访问标记
            frontier = torch.nonzero(new_frontier, as_tuple=False).squeeze(-1).to(ligand_pos.device)
        
        # 获取被访问的节点索引
        selected_local = torch.nonzero(visited_mask, as_tuple=False).squeeze(-1)
        selected_atoms = mol_atom_idx[selected_local]
        mask[selected_atoms] = True

    return mask


def keep_scaffold_and_groups(mol, scaffold, attachment_atoms):
    """
    保留小分子中的骨架和与指定原子相连的基团
    :param mol: 需要处理的小分子
    :param scaffold: 骨架分子
    :param attachment_atoms: 需要保留的基团连接的原子索引列表
    :return: 处理后的小分子
    """

    # scaffold_matches = mol.GetSubstructMatch(scaffold)
    # atoms_to_keep = set(scaffold_matches)

    # atoms_to_keep = [i for i in range(scaffold.GetNumAtoms())]
    # scaffold_atom_num = scaffold.GetNumAtoms()

    confA = mol.GetConformer()
    confB = scaffold.GetConformer()
    coordsA = [confA.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    coordsB = [confB.GetAtomPosition(j) for j in range(scaffold.GetNumAtoms())]

    atoms_to_keep = []
    mol2scaffold = {}
    for i, posA in enumerate(coordsA):
        for j, posB in enumerate(coordsB):
            if np.linalg.norm(posA - posB) < 1e-3:
                atoms_to_keep.append(i)
                mol2scaffold[i] = j
                break  # 一旦匹配，就不用继续比对 B 的其他原子
    scaffold_atom_num = scaffold.GetNumAtoms()

    # print('atoms_to_keep',atoms_to_keep)

    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.Kekulize(scaffold, clearAromaticFlags=True)

    while(len(attachment_atoms) != 0):
        attachment_atoms_new = set()
        atoms_to_keep_new = set([i for i in atoms_to_keep])
        for idx in attachment_atoms:
            # 查找与指定原子直接相连的原子，并将它们标记为保留
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                atoms_to_keep_new.add(neighbor.GetIdx())
                if neighbor.GetIdx() not in atoms_to_keep:
                    attachment_atoms_new.add(neighbor.GetIdx())
                    # print('attachment_atoms_new', attachment_atoms_new, idx)
        attachment_atoms = attachment_atoms_new
        atoms_to_keep = atoms_to_keep_new
    # print('atoms_to_keep',atoms_to_keep)
    
    # 调整键的类型
    emol = Chem.RWMol(mol)
    for begin_atom_idx in range(emol.GetNumAtoms()):
        for end_atom_idx in range(begin_atom_idx + 1, emol.GetNumAtoms()):

            if begin_atom_idx in mol2scaffold.keys() and end_atom_idx in mol2scaffold.keys():
                bond_old = scaffold.GetBondBetweenAtoms(mol2scaffold[begin_atom_idx], mol2scaffold[end_atom_idx])
                # bond.SetIsAromatic(bond_old.GetIsAromatic())
                if bond_old:
                    # print(bond_old.GetBondType())
                    bond = emol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
                    if not bond:
                        emol.AddBond(begin_atom_idx, end_atom_idx, bond_old.GetBondType())
                        bond = emol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
                        # print(bond_old.GetBondType(), bond.GetBondType())
                    else:
                        bond.SetBondType(bond_old.GetBondType())

    # Draw.MolToFile(emol, 'test2.png', size=(1000,1000))
    # writer = Chem.SDWriter('test2.sdf')
    # writer.write(emol)
    # writer.close()

    # 使用EditMol删除未标记的原子
    for atom_idx in reversed(range(mol.GetNumAtoms())):
        if atom_idx not in atoms_to_keep:
            emol.RemoveAtom(atom_idx)

    # Draw.MolToFile(emol, 'test3.png', size=(1000,1000))
    # writer = Chem.SDWriter('test3.sdf')
    # writer.write(emol)
    # writer.close()

    try:
        Chem.SanitizeMol(emol)
        return emol.GetMol()
    except:
        return None


def process_sdf_files(molecule_sdf, scaffold, attachment_atoms):
    """
    处理SDF文件,保留骨架和指定基团
    :param scaffold_sdf: 骨架SDF文件
    :param molecule_sdf: 需要处理的小分子SDF文件
    :param attachment_atoms: 基团连接的原子索引列表
    """

    molecule_supplier = Chem.SDMolSupplier(molecule_sdf)
    
    mol = molecule_supplier[0]
    mol = Chem.RemoveHs(mol)
    # 保留骨架和基团
    new_mol = keep_scaffold_and_groups(mol, scaffold, attachment_atoms)

    # img = Chem.Draw.MolToImage(new_mol)
    # img.show()

    # writer = Chem.SDWriter(output_sdf)
    # writer.write(new_mol)
    # writer.close()
    return new_mol

def modify_scaffold(gen_dir, scaffold_mol, attachment_atoms, min_add_num, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    scaffold_atom_num = scaffold_mol.GetNumAtoms()
    SMILES_list = []

    for file in os.listdir(gen_dir):
        if file.endswith('.sdf'):
            molecule_sdf = os.path.join(gen_dir, file)
            new_mol = process_sdf_files(molecule_sdf, scaffold_mol, attachment_atoms)
            if not new_mol:
                continue
            if new_mol.GetNumAtoms() >= scaffold_atom_num + min_add_num:
                print(molecule_sdf)
                SMILES = Chem.MolToSmiles(new_mol)
                if SMILES not in SMILES_list:
                    SMILES_list.append(SMILES)

                    # Draw.MolToFile(new_mol, os.path.join(output_dir, f"filter_{count}.png"))

                    try:
                        output_file = os.path.join(output_dir, f"filter_{count+1}.sdf")
                        with Chem.SDWriter(output_file) as f:
                            f.write(new_mol)
                        count += 1
                        print(count)
                    except:
                        continue
    print(f'Get {count} mols!')

if __name__ == '__main__':
    data_dir = "."
    gen_dir = 'output_test3'
    scaffold_sdf = '7rbt_scaffold.sdf'
    attachment_atoms = [2,10,21,30]
    min_add_num = 7
    output_dir = os.path.join(gen_dir, 'frag_part_filter')
    
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    scaffold = Chem.SDMolSupplier(scaffold_sdf)[0]
    scaffold = Chem.RemoveHs(scaffold)
    scaffold_atom_num = scaffold.GetNumAtoms()
    SMILES_list = []

    fdit = os.path.join(data_dir, gen_dir)
    for file in os.listdir(fdit):

        # if file != '97.sdf':
        #     continue

        if file.endswith('.sdf'):
            molecule_sdf = os.path.join(fdit, file)
            new_mol = process_sdf_files(molecule_sdf, scaffold_sdf, attachment_atoms)
            if not new_mol:
                continue
            if new_mol.GetNumAtoms() >= scaffold_atom_num + min_add_num:
                print(molecule_sdf)
                SMILES = Chem.MolToSmiles(new_mol)
                if SMILES not in SMILES_list:
                    SMILES_list.append(SMILES)

                    # Draw.MolToFile(new_mol, os.path.join(output_dir, f"filter_{count}.png"))

                    try:
                        output_file = os.path.join(output_dir, f"filter_{count+1}.sdf")
                        with Chem.SDWriter(output_file) as f:
                            f.write(new_mol)
                        count += 1
                        print(count)
                    except:
                        continue
    print(f'Get {count} mols!')