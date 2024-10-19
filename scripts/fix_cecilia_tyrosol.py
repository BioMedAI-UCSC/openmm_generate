#!/usr/bin/env python3
import subprocess
import mdtraj
import re
import glob
import os
import itertools

# openmm templates fail to match cecilia pdb files since they used charmm
# https://github.com/torchmd/torchmd-protein-thermodynamics/tree/main/Datasets
# remove ivalid atoms on Tyrosine so it can match it

def fix_TYR(input_path: str, output_path: str):
    traj = mdtraj.load(input_path, standard_names=False)

    # first_residue = traj.topology.residue(0)
    # last_residue = traj.topology.residue(1)

    atoms_remove = []
    
    name_pattern = re.compile("(.*(T|Y)[0-9]*)|(H[0-9]+)")
    
    for atom in traj.topology.atoms: #TODO: this should only run on the first and last residues
        if atom.name == "OXT": #daniel said this is special --andy
            continue
        if name_pattern.fullmatch(atom.name):
            atoms_remove.append(atom.index)
    
    atoms_to_keep = [a.index for a in traj.topology.atoms if a.index not in atoms_remove]

    modified = mdtraj.load_pdb(input_path, top=traj.top, atom_indices=atoms_to_keep)
    modified.save_pdb(output_path)

OUT_PATH = "/media/DATA_18_TB_1/andy/tica_sampled_starting_poses/wwdomain1_TYR_fixed"

subprocess.run(["rm", "-r", OUT_PATH])
subprocess.run(["mkdir", OUT_PATH])
for pdb in glob.glob('/media/DATA_18_TB_1/andy/tica_sampled_starting_poses/wwdomain1/*.pdb'):
    print("DOING", pdb)
    basename = os.path.basename(pdb)
    fix_TYR(pdb, os.path.join(OUT_PATH, basename))
