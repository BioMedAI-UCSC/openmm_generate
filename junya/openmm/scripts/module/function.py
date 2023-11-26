from openmm.app.pdbfile import PDBFile
import numpy as np
import os


class ProteinAtom():
    """
    A class representing protein atoms and providing methods for extracting atom lines and getting atom indices.
    """

    def __init__(self, input_path, output_path) -> None:
        """
        Initialize the ProteinAtom object.

        Parameters:
        - input_path (str): The path to the input file.
        - output_path (str): The path to the output file.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.indices = None
        self.check = 0

    def extract_atom(self):
        """
        Extract the ATOM lines from the input file and write them to the output file.
        Note that the ATOM lines in PDB file mean the protein atoms' lines. 
        Extracting the ATOM lines is necessary for getting the atom indices from protein.
        
        FIXME: Currently, it reads the pdb file directly and extracts the indices of atoms belonging to proteins, but this is not the best way. Is it possible to do this with the Openmm function?

        """
        atom_lines = []
        with open(self.input_path, 'r') as input_file:
            for line in input_file:
                # extract only ATOM lines
                if line.startswith('ATOM'):
                    atom_lines.append(line)
                    self.check += 1

        with open(self.output_path, 'w') as output_file:
            output_file.writelines(atom_lines)

    def get_indices(self):
        """
        Get the atom indices from the output file.

        Returns:
        - indices (numpy.ndarray): An array of atom indices from protein.
        """
        pdb = PDBFile(self.output_path)
        topology = pdb.getTopology()
        self.indices = np.array([atom.index for atom in topology.atoms()])
        assert self.check == len(self.indices), \
        "The number of Atom Lines is not equal to the number of atoms in the output file."
        
        return self.indices


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

