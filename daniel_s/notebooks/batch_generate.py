#!/usr/bin/env python3
import sys
sys.path.append("../../junya/openmm/scripts")

import os
import time
import traceback
from module import function
from module import preprocess
from module import simulation
from openmm.app import PDBFile

# Log capture
# From https://stackoverflow.com/questions/1218933/can-i-redirect-the-stdout-into-some-sort-of-string-buffer
import sys
from io import StringIO

class RedirectOutputs:
    def __init__(self):
        self._stdout = None
        self._stderr = None
        self._string_io = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = sys.stderr = self._string_io = StringIO()
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def __str__(self):
        return self._string_io.getvalue()

def prepare_one(pdbid, data_dir=None):
    if data_dir:
        function.set_data_dir(data_dir)
    if os.path.exists(function.get_data_path(f'{pdbid}/processed/finished.txt')):
        return
    print("Processing", pdbid)

    t0 = time.time()
    ok = True
    with RedirectOutputs() as log:
        try:
            preprocess.prepare_protein(pdbid)
        except Exception as e:
            ok = False
            print(e)
    with open(function.get_data_path(f'{pdbid}/processed/{pdbid}_process.log'),"wb") as f:
        f.write(str(log).encode("utf-8"))
    t1 = time.time() - t0
    finished_str = f"{pdbid} {('error', 'ok')[int(ok)]} ({round(t1,4)} seconds)"
    with open(function.get_data_path(f'{pdbid}/processed/finished.txt'), "w", encoding="utf-8") as finished_file:
        finished_file.write(finished_str)
    print(" ", finished_str)

def simulate_one(pdbid, data_dir=None, steps=10000, report_steps=1):
    if data_dir:
        function.set_data_dir(data_dir)
    finished_file_path = function.get_data_path(f'{pdbid}/simulation/finished.txt')
    if os.path.exists(finished_file_path):
        os.remove(finished_file_path)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("Simulating", pdbid, "on gpu", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Simulating", pdbid)

    t0 = time.time()
    ok = True
    with RedirectOutputs() as log:
        try:
            pdb_path = function.get_data_path(f'{pdbid}/processed/{pdbid}_processed.pdb')
            atom_indices = function.get_non_water_atom_indexes(PDBFile(pdb_path).getTopology())
            simulation.run(pdbid, pdb_path, steps, report_steps=report_steps, atomSubset=atom_indices)
        except Exception as e:
            ok = False
            traceback.print_tb(e.__traceback__)
    with open(function.get_data_path(f'{pdbid}/simulation/{pdbid}_simulation.log'),"wb") as f:
        f.write(str(log).encode("utf-8"))
    t1 = time.time() - t0
    finished_str = f"{pdbid} {('error', 'ok')[int(ok)]} ({round(t1,4)} seconds)"
    with open(finished_file_path, "w", encoding="utf-8") as finished_file:
        finished_file.write(finished_str)
    print(" ", finished_str)

def init_on_gpu(gpu_list, counter):
    gpu_id = None
    with counter.get_lock():
        gpu_id = gpu_list[counter.value % len(gpu_list)]
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

def main():
    import json
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument("pdbid_list", type=str, help="A json file containing an array of PDB ids to process")
    parser.add_argument("--batch-index", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--pool-size", default=10, type=int, help="Number of simultaneous simulations to run")
    parser.add_argument("--steps", default=10000, type=int, help="Total number of steps to run")
    parser.add_argument("--report-steps", default=1, type=int, help="Save data every n-frames")
    parser.add_argument("--data-dir", default="../data/", type=str)
    parser.add_argument("--gpus", default=None, type=str, help="A comma delimited lists of GPUs to use e.g. '0,1,2,3'")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.data_dir):
        print("Invalid data directory:", args.data_dir)
        return 1

    # if args.batch_size is None or args.batch_index is None:
    #     print("Batch size and index must both be set.")
    #     parser.print_usage()

    try:
        multiprocessing.set_start_method('spawn') # because NERSC says to use this one?
    except Exception as e:
        print("Multiprocessing:", e)

    with open(args.pdbid_list,"r") as f:
        pdbid_list = json.load(f)
    batch_pdbid_list = pdbid_list[args.batch_index*args.batch_size:(args.batch_index+1)*args.batch_size]
    print(batch_pdbid_list)

    init_function = None
    init_args = None
    if args.gpus is not None:
        gpu_list = [int(i) for i in args.gpus.split(",")]
        init_args = (gpu_list, multiprocessing.Value('i', 0, lock=True))
        init_function = init_on_gpu

    t0 = time.time()
    with multiprocessing.Pool(args.pool_size, initializer=init_function, initargs=init_args) as pool:
        pending_results = []
        for pdbid in batch_pdbid_list:
            pending_results += [pool.apply_async(simulate_one, (pdbid, args.data_dir, args.steps, args.report_steps))]
        
        while pending_results:
            pending_results = [i for i in pending_results if not i.ready()]
            if pending_results:
                pending_results[0].wait(1)
    
    t1 = time.time() - t0
    print(f"Finished {len(batch_pdbid_list)} in {round(t1,4)} seconds")

    return 0

if __name__ == "__main__":
    sys.exit(main())
