from openmm.app.pdbfile import PDBFile
import numpy as np
import os




def create_folder(folder_path):
    """
    Create a folder if it does not exist, or do nothing if it already exists.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            os.makedirs(f"{folder_path}/raw")
            os.makedirs(f"{folder_path}/interim")
            os.makedirs(f"{folder_path}/processed")
            os.makedirs(f"{folder_path}/result")
            os.makedirs(f"{folder_path}/simulation")
            
            print(f"Folder created: {folder_path}")
        except OSError as e:
            print(f"Error: Unable to create folder {folder_path}. {e}")
    else:
        print(f"Folder already exists: {folder_path}")




def get_atomSubset(pdb_path=str):
    """
    Get the subset of atom indices for protein residues in a PDB file.
    
    Args:
        pdb_path (str): The path to the PDB file.
        
    Returns:
        list: A list of atom indices corresponding to protein residues.
    """
    
    proteinResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR', 'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL']
    
    pdb = PDBFile(pdb_path)

    atomSubset = []
    topology = pdb.getTopology()
    for atom in topology.atoms():
        if atom.residue.name in proteinResidues:
            atomSubset.append(atom.index)
    
    return atomSubset