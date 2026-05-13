import os
import argparse
from rdkit import Chem
from rdkit.Chem import Draw


def main():
    parser = argparse.ArgumentParser(
        description="Visualize molecule from SDF and save atom-index figure."
    )
    parser.add_argument(
        "--sdf",
        type=str,
        required=True,
        help="Path to input SDF file"
    )

    args = parser.parse_args()

    sdf_path = args.sdf

    if not os.path.exists(sdf_path):
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    # output directory = same directory as sdf
    out_dir = os.path.dirname(os.path.abspath(sdf_path))

    # output png name
    base_name = os.path.splitext(os.path.basename(sdf_path))[0]
    out_png = os.path.join(out_dir, f"{base_name}.png")

    # load molecule
    mol = Chem.SDMolSupplier(sdf_path)[0]

    if mol is None:
        raise ValueError(f"Failed to read molecule from: {sdf_path}")

    smiles = Chem.MolToSmiles(mol)

    print(f"SDF file: {sdf_path}")
    print(f"SMILES: {smiles}")

    # remove conformers for 2D depiction
    mol.RemoveAllConformers()

    # add atom indices
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("molAtomMapNumber", str(i))

    # print atom index string
    atoms_index = list(range(mol.GetNumAtoms()))
    atoms_index_str = " ".join(map(str, atoms_index))

    print("atoms_index_str =")
    print(atoms_index_str)

    # save figure
    Draw.MolToFile(
        mol,
        out_png,
        size=(1000, 1000)
    )

    print(f"Saved figure to: {out_png}")


if __name__ == "__main__":
    main()