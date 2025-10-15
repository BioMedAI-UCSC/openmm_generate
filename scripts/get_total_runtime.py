#!/usr/bin/env python3
from pathlib import Path

root = Path("/media/DATA_18_TB_1/andy/benchmark_set_5/")



proteins = [
    "a3d",
    "bba",
    "chignolin",
    "homeodomain",
    "lambda",
    "proteinb",
    "proteing",
    "trpcage",
    "wwdomain"]

def get_traj_time(traj_path: Path) -> float:
    name = traj_path.parts[-1]
    finished_path = traj_path.joinpath("simulation/finished.txt")
    if finished_path.exists():
        with open(finished_path, "r") as f:
            data = f.read()
            assert data[:len(name)] == name
            rest = data[len(name):]
            assert rest[:5] == " ok ("
            assert rest[-9:] == " seconds)"
            thing = rest[5:-9]
            return float(thing)
    else:
        print(f"path {finished_path} mising!!")
        return 0


for protein in proteins:
    stuff = root.joinpath(protein)
    dirs: list[Path] = list(stuff.iterdir())
    times = [get_traj_time(x) for x in dirs]
    total_secs = sum(times)
    print(f"{protein}: {total_secs} seconds of total node simulation time")


