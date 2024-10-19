# For making starting positions for batch_generate.py
# at https://github.com/BioMedAI-UCSC/openmm_generate

import glob
import deeptime
import mdtraj
from numpy.typing import NDArray
import numpy as np
import itertools
from matplotlib import pyplot as plt
import os
import json


NUM_STARTING = 100000000
DIST_THRESH = 0.0001

def make_tica_model(bond_lens):
    lagtime=20
    estimator = deeptime.decomposition.TICA(lagtime=lagtime, dim=None)
    
    bond_lens = [x for x in bond_lens if (x.shape[0] > lagtime)]
    print("fitting tica model")
    for i, (X, Y) in enumerate(deeptime.util.data.timeshifted_split(bond_lens, lagtime=lagtime)):
        print(f"fitted {i}/{len(bond_lens)} trajectories")
        estimator.partial_fit((X, Y))
    model_onedim = estimator.fetch_model()
    print("done fitting tica model")
    return model_onedim

def get_bonds(top: mdtraj.Topology) -> list[tuple[int, int]]:
    return list(map(lambda x: (x[0].index, x[1].index), top.bonds))

def calc_bond_lens(traj: mdtraj.Trajectory, pairs: list[tuple[int, int]]) -> NDArray:
    distances: NDArray = mdtraj.compute_distances(traj, pairs)
    return distances

def make_starting_pos(top: mdtraj.Topology, trajs: list[mdtraj.Trajectory], do_plot=False):
    print("calculating bond lens")

    bonds = get_bonds(top)
    pairs = list(itertools.combinations(range(0, trajs[0].n_atoms), 2) if(len(bonds) == 0) else bonds)
    bond_lens = list(map(lambda x: calc_bond_lens(x, pairs), trajs))
    print("done calculating bond lens")
    tica_model = make_tica_model(bond_lens)
    projected_datas = list(map(lambda x: tica_model.transform(x), bond_lens))

    datas = np.concatenate(projected_datas)[:, :2]
    datas_index = np.concatenate(
        [np.array(
            [np.repeat([traj_num], traj.n_frames), np.arange(traj.n_frames)]).transpose()
         for traj_num, traj in enumerate(trajs)])
    assert(datas_index.shape[0] == datas.shape[0])
    starting_positions = []

    while len(datas) > 0:
        
        print(f"found {len(starting_positions)} points, remaining {len(datas)} points")
        choose_index = 0 #choose some random index
        point = datas[choose_index]
        def dist_squared(p1, p2):
            return ((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2)

        starting_positions.append((point, datas_index[choose_index]))
        to_remove = []
        for i, a in enumerate(datas):
            if dist_squared(a, point) <= DIST_THRESH:
                to_remove.append(i)
        datas = np.delete(datas, to_remove, axis=0)
        datas_index = np.delete(datas_index, to_remove, axis=0)
        assert(datas_index.shape[0] == datas.shape[0])

    starting_positions_pos = np.array(list(map(lambda x: x[0], starting_positions)))
    if do_plot:
        fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True)
        axs[0][0].set_xlabel("TICA 0th component")
        axs[0][0].set_ylabel("TICA 1st component")

        for i, projected_data in enumerate(projected_datas):
            num_show = min(projected_data.shape[0], NUM_STARTING)
            axs[0][0].scatter(projected_data[:num_show, 0], projected_data[:num_show, 1], s=2, c="blue")
            axs[0][0].scatter(starting_positions_pos[:, 0], starting_positions_pos[:, 1], s=2, c="red")
    
        fig.savefig("test.png", format='png')
        plt.close()
        
    print(f"saving {len(starting_positions)} starting positions")
    with open("points.json", "w") as f:
        f.write(json.dumps(starting_positions_pos.tolist(), indent=4))
    
    for i, (traj_num, frame_num) in enumerate(map(lambda x: x[1], starting_positions)):
        frame = trajs[traj_num][frame_num]
        assert frame.xyz.shape[0] == 1
        traj = mdtraj.Trajectory(frame.xyz, top)
        traj.save(f"output/starting_pos_{i}.pdb")

def load_native_trajs(native_trajs_dir: str) -> tuple[mdtraj.Topology, list[mdtraj.Trajectory]]:
    paths = glob.glob(os.path.join(native_trajs_dir, "filtered/*/*.xtc"))
    print(f"loading {len(paths)} trajectories")
    top = os.path.join(native_trajs_dir, "filtered/filtered.pdb")
    
    trajs = list(map(lambda p: load_native_path(p, top), paths))

    topology = mdtraj.load(top).top

    print("done loading trajectories")
    return topology, trajs

def load_native_path(native_path: str, pdb) -> mdtraj.Trajectory:
    out = mdtraj.load_xtc(native_path, top=pdb)
    return out


def main():
    top, trajs = load_native_trajs("/media/DATA_18_TB_1/andy/torchmd_data/trajectories/wwdomain")
    make_starting_pos(top, trajs, do_plot=True)

if __name__ == "__main__":
    main()
