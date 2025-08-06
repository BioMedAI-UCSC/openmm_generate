#!/usr/bin/env python3
import argparse
from pathlib import Path
import mdtraj
import numpy
import itertools


def main():
    parser = argparse.ArgumentParser(description="Calculate Gelman-Rubin statistic of trajectories",)
    parser.add_argument("traj_folder", type=Path)
    args = parser.parse_args()

    folder: Path = args.traj_folder

    paths: list[Path] = []
    for finished_path in folder.glob("*/processed/finished.txt"):
        stuff = finished_path.read_text()
        if "ok" in stuff:
            paths.append(Path(finished_path.parents[1]))
        else:
            print(f"path {finished_path} broke or something")


    chain_means: list[numpy.typing.NDArray] = []
    chain_variances: list[numpy.typing.NDArray] = []

    paths = paths
        
    for i, path in enumerate(paths):
        print(f"loaded {i}/{len(paths)} paths")
        h5_path = list(path.glob("result/*.h5"))
        assert len(h5_path) == 1
        traj = mdtraj.load_hdf5(h5_path[0])
        assert traj.topology is not None
        ca_atoms = traj.topology.select('name CA')
        traj_ca = traj.atom_slice(ca_atoms)


        assert traj_ca.xyz is not None
        pairs = list(itertools.combinations(range(0, traj_ca.n_atoms), 2))
        coords: numpy.typing.NDArray = mdtraj.compute_distances(traj_ca, pairs)
        first_10_percent = int(coords.shape[0] * 0.1)
        coords = coords[first_10_percent:]
        chain_mean = numpy.mean(coords, axis=0)
        chain_variance = numpy.var(coords, axis=0, ddof=0)

        chain_means.append(chain_mean)
        chain_variances.append(chain_variance)

    means: numpy.typing.NDArray = numpy.stack(chain_means)
    variances: numpy.typing.NDArray = numpy.stack(chain_variances)

    means_mean = numpy.mean(means, axis=0)
    means_variance = numpy.var(means, axis=0)
    variances_mean = numpy.mean(variances, axis=0)
    
    
    gelman_rubin = (variances_mean + means_variance) / variances_mean
    print(gelman_rubin)


if __name__ == "__main__":
    main()
