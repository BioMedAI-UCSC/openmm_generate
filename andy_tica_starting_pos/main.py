from random import randrange
import scipy
import glob
import deeptime
import mdtraj
from numpy.typing import NDArray
import numpy as np
import itertools
from matplotlib import pyplot as plt
import sys
import os
import json
import tempfile

sys.path.insert(0, "/home/andy/cgschnet/cgschnet/scripts/")#to import preproccess.py
prior_params = json.load(open(os.path.join("/home/andy/cgschnet/cgschnet/data/result-2024.08.21-10.16.28/", "prior_params.json"), "r"))

NUM_STARTING = 100000000

def make_tica_model(bond_lens):
    lagtime=20
    estimator = deeptime.decomposition.TICA(lagtime=lagtime, dim=None)
    
    bond_lens = [x for x in bond_lens if (x.shape[0] > lagtime)]

    for X, Y in deeptime.util.data.timeshifted_split(bond_lens, lagtime=lagtime):
        estimator.partial_fit((X, Y))
    model_onedim = estimator.fetch_model()
    return model_onedim

def get_bonds(traj: mdtraj.Trajectory) -> list[tuple[int, int]]:
    return list(map(lambda x: (x[0].index, x[1].index), traj.topology.bonds))

def calc_bond_lens(traj: mdtraj.Trajectory) -> NDArray:
    bonds = get_bonds(traj)

    pairs = itertools.combinations(range(0, traj.n_atoms), 2) if(len(bonds) == 0) else bonds
    pairs = list(pairs)
    distances: NDArray = mdtraj.compute_distances(traj, pairs)
    return distances

def make_plot(trajs: list[mdtraj.Trajectory]):
    bond_lens = list(map(lambda x: calc_bond_lens(x), trajs))
    tica_model = make_tica_model(bond_lens)
    projected_datas = list(map(lambda x: tica_model.transform(x), bond_lens))
    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True)
    axs[0][0].set_xlabel("TICA 0th component")
    axs[0][0].set_ylabel("TICA 1st component")

    datas = np.concatenate(projected_datas).transpose()[:2, :]
    
    for i, projected_data in enumerate(projected_datas):
        num_show = min(projected_data.shape[0], NUM_STARTING)
        axs[0][0].scatter(projected_data[:num_show, 0], projected_data[:num_show, 1], s=2, c="blue")

    datas = np.concatenate(projected_datas)[:, :2]
    print(datas.shape)
    starting_positions = []

    while len(datas) > 0:
        print("len =", len(starting_positions), "remaining", len(datas))
        index = randrange(len(datas))
        point = datas[index]
        def dist_squared(p1, p2):
            return ((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2)

        DIST_THRESH = 0.000001
        
        starting_positions.append(point)
        to_remove = []
        for i, a in enumerate(datas):
            if dist_squared(a, point) <= DIST_THRESH:
                to_remove.append(i)
        datas = np.delete(datas, to_remove, axis=0)


    starting_positions = np.array(starting_positions)
    axs[0][0].scatter(starting_positions[:, 0], starting_positions[:, 1], s=2, c="red")
    
    
    fig.savefig("poo.png", format='png')

def load_native_trajs(native_trajs_dir: str) -> list[mdtraj.Trajectory]:
    import preprocess
    prior_name = prior_params["prior_configuration_name"]
    prior_builder = preprocess.prior_types[prior_name]()
    top = os.path.join(native_trajs_dir, "extract/filtered/filtered.pdb")
    mol = prior_builder.write_psf(top, None)

    with tempfile.TemporaryDirectory() as tmpdirname:
        topology_path = os.path.join(tmpdirname, "topology.pdb")
        mol.write(topology_path)
        topology = mdtraj.load(topology_path).top
        atoms_idx = prior_builder.select_atoms(topology)
        
        paths = glob.glob(os.path.join(native_trajs_dir, "extract/filtered/*/*.xtc"))# [:100]
        trajs = list(map(lambda p: load_native_path(p, atoms_idx, topology, top), paths))
        return trajs

def load_native_path(native_path: str, atoms_idx, topology, pdb) -> mdtraj.Trajectory:
    out = apply_cg(mdtraj.load_xtc(native_path, top=pdb), atoms_idx)
    out.top = topology
    return out

def apply_cg(traj, atom_indicies):
    new_traj = traj.atom_slice(atom_indicies)
    assert new_traj.n_atoms == len(atom_indicies)
    return new_traj


def main():
    thing = load_native_trajs("/media/DATA_18_TB_2/andy/benchmark_set_2/trajectory_datas/chignolin")
    make_plot(thing)

if __name__ == "__main__":
    main()
