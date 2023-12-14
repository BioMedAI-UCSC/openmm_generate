import openmm as mm
import openmm.app as app
from openmm import unit
import pdbfixer
import requests
import json
from module import ligands
from module import function


def prepare_protein(pdbid=str, remove_ligands=False):
    """
    Preprocesses a protein by downloading the PDB file, fixing missing residues and atoms,
    adding missing hydrogens, adding solvent, and writing the processed PDB file.

    Args:
        pdbid (str): The PDB ID of the protein.

    Returns:
        None
    """
    
    print(f"Preprocess of {pdbid}")
    # create folder
    function.create_folder(f"../data/{pdbid}")
    pdb_path = f"../data/{pdbid}/raw/{pdbid}.pdb"
    pdb_url = f"https://files.rcsb.org/download/{pdbid}.pdb"

    # download pdb file
    r = requests.get(pdb_url)
    r.raise_for_status()
    with open(pdb_path, "wb") as f:
        f.write(r.content)

    fixer = pdbfixer.PDBFixer(pdb_path)

    # find missing residues and atoms
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    print(f"Missing residues: {fixer.missingResidues}")
    print(f"Missing terminals: {fixer.missingTerminals}")
    print(f"Missing atoms: {fixer.missingAtoms}")

    # remove missing residues at the terminal
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        # terminal residues
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]

    # check if the terminal residues are removed
    for key in list(keys):
        chain = chains[key[0]]
        assert key[1] != 0 or key[1] != len(list(chain.residues())), "Terminal residues are not removed."

    # find nonstandard residues
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    # add missing atoms, residues, and terminals
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # add missing hydrogens
    ph = 7.0
    fixer.addMissingHydrogens(ph)

    # make modeller
    modeller = app.Modeller(fixer.topology, fixer.positions)

    # add ligands back to the prepaired protein
    small_molecules = ligands.replace_ligands(pdb_path, modeller, remove_ligands=remove_ligands)

    print("\nAfter the process")
    print(f"Missing residues: {fixer.missingResidues}")
    print(f"Missing terminals: {fixer.missingTerminals}")
    print(f"Missing atoms: {fixer.missingAtoms}")

    # set the forcefield
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    if small_molecules:
        json.dump(small_molecules, open(f'../data/{pdbid}/processed/{pdbid}_processed_ligands_smiles.json', 'w'))
        template_cache_path = f'../data/{pdbid}/processed/{pdbid}_processed_ligands_cache.json'
        ligands.add_ff_template_generator_from_smiles(forcefield, small_molecules, cache_path=template_cache_path)

    # Small molecules we've added templates for will be named "UNK"
    unmatched_residues = [r for r in forcefield.getUnmatchedResidues(modeller.topology) if r.name != "UNK"]
    if unmatched_residues:
        raise RuntimeError("Structure still contains unmatched residues after fixup: " + str(unmatched_residues))

    # solvent model: tip3p. Default is NaCl
    modeller.addSolvent(forcefield, padding=1.0 * unit.nanometers, ionicStrength=0.15 * unit.molar)

    # write the processed pdb file & ligand templates
    top = modeller.getTopology()
    pos = modeller.getPositions()
    app.PDBFile.writeFile(top, pos, open(f'../data/{pdbid}/processed/{pdbid}_processed.pdb', 'w'))
