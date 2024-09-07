import mdtraj

atoms_remove = set(map(lambda x: x-1, [1, 2, 3, 4, 5, 6, 170, 171, 172, 173, 174, 175]))

traj = mdtraj.load(f"/media/DATA_18_TB_2/andy/benchmark_generate_input/chignolin.pdb")

for atom in traj.topology.atoms:
    print(atom.name, atom.index)

atoms_to_keep = [a.index for a in traj.topology.atoms if a.index not in atoms_remove]

modified = mdtraj.load_pdb("/media/DATA_18_TB_2/andy/benchmark_generate_input/chignolin.pdb", top=traj.top, atom_indices=atoms_to_keep)

modified.save_pdb("./test.pdb")
