# OpenMM Generate

A Python-based molecular dynamics simulation toolkit using OpenMM for generating all-atom protein trajectories at scale.

## Overview

This repository provides a module for generating standard molecular dynamics simulations using OpenMM for all-atom protein representations. It is part of a larger pipeline for molecular dynamics and machine learning research, designed to run efficiently on GPU-accelerated HPC systems.

## Status

⚠️ **IMPORTANT**: This code is currently being ported and refactored from private repositories for public release. Full documentation and tutorials will be provided within 1-2 weeks.

## Features

- **Automated Protein Preparation**: Download PDB files, add missing residues/atoms, solvate systems
- **Ligand Parameterization**: Automatic parameterization of small molecules using GAFF
- **Flexible Simulation Options**: Configurable timesteps, temperatures, and integrators
- **Efficient Data Storage**: HDF5-based trajectory storage with forces and energies
- **GPU Acceleration**: Multi-GPU support with automatic distribution
- **Resumable Simulations**: Checkpoint-based continuation of interrupted runs
- **Batch Processing**: Process multiple proteins in parallel across GPUs
- **HPC Integration**: SLURM scripts for Perlmutter, Delta, and other HPC systems

## Installation

### Prerequisites

- Python 3.11 or 3.12
- CUDA 12.1+ (for GPU acceleration)
- Conda or Mamba package manager

### Environment Setup

Create the conda environment using the provided configuration:

```bash
conda env create -f conda_envs/delta.yml -n openmm_generate
conda activate openmm_generate
```

Or for a minimal setup:

```bash
conda env create -f conda/openmm_cu121_py312.yml -n openmm
conda activate openmm
```

### Dependencies

Key dependencies include:
- OpenMM 8.1+
- OpenMM Forcefields
- OpenFF Toolkit
- PDBFixer
- RDKit
- MDTraj
- h5py

## Quick Start

### Basic Usage

Prepare and simulate a single protein:

```bash
cd scripts
python batch_generate.py <pdb_id> --prepare --steps 10000 --report-steps 10
```

### Example Commands

**Run a single PDB for 1000 frames, removing ligands:**
```bash
./batch_generate.py 1ABC --prepare --remove-ligands --data-dir=../data --steps 1000
```

**Run all PDBs from a JSON file for 10,000 frames with ligands:**
```bash
./batch_generate.py openmm_2023.12.20_N1.json --prepare --data-dir=../data --steps 10000
```

**Run with custom integrator (4fs timestep at 350K):**
```bash
./batch_generate.py pdb_list.json --prepare --data-dir=../data --steps 10000 \
  --integrator=integrator_350k_4fs.json
```

**Process local PDB files from a directory:**
```bash
./batch_generate.py --prepare --steps 10000 --report-steps=10 \
  --input-dir=../data/input/ --data-dir=../data/output
```

**Multi-GPU parallel processing:**
```bash
./batch_generate.py pdb_list.json --prepare --steps 100000 --report-steps=100 \
  --gpus=0,1,2,3 --pool-size=8 --data-dir=../data
```

## Command Line Options

### Core Options

- `pdbid_list`: PDB IDs to process, or path to JSON file with PDB ID array
- `--prepare`: Run preparation (add solvent, fix residues) if not already done
- `--prepare-implicit`: Use implicit solvent model instead of explicit
- `--remove-ligands`: Remove ligands instead of parameterizing them
- `--steps`: Total number of simulation steps (default: 10000)
- `--report-steps`: Save data every n frames (default: 1)
- `--data-dir`: Output data directory (default: ../data/)
- `--input-dir`: Input directory for local PDB files

### Advanced Options

- `--integrator`: Path to JSON file with integrator parameters (dt, temperature, etc.)
- `--gpus`: Comma-separated list of GPU IDs to use (e.g., "0,1,2,3")
- `--pool-size`: Number of simultaneous simulations (default: 10)
- `--batch-size`: Split input list into batches of this size
- `--batch-index`: Select which batch to run (for array jobs)
- `--timeout`: Maximum runtime in hours (e.g., 2.5 for 2.5 hours)
- `--force`: Force re-run of completed simulations

## Integrator Configuration

Custom integrator parameters can be specified via JSON files:

**integrator_4fs.json** (4 femtosecond timestep):
```json
{
    "dt": "0.004*picoseconds"
}
```

**integrator_350k_4fs.json** (4fs timestep at 350K):
```json
{
    "dt": "0.004*picoseconds",
    "temperature": "350*kelvin"
}
```

Available parameters:
- `dt`: Timestep (default: 0.002 ps)
- `temperature`: Temperature (default: 300 K)
- `friction`: Friction coefficient (default: 1.0/ps)
- `pressure`: Pressure for barostat (default: 1.0 atm)
- `barostatInterval`: Barostat update frequency (default: 25 steps)

## HPC Usage

### SLURM Job Submission

The repository includes SLURM scripts for various HPC systems:

**Perlmutter (NERSC):**
```bash
sbatch perlmutter1x4_N1000.slurm
```

**Delta (NCSA):**
```bash
sbatch delta.slurm
```

### Example SLURM Configuration

```bash
#!/bin/bash
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00

export OMP_NUM_THREADS=32

srun python3 batch_generate.py pdb_list.json \
  --batch-index $SLURM_ARRAY_TASK_ID \
  --batch-size 25 \
  --pool-size 8 \
  --gpus="0,1,2,3" \
  --prepare \
  --steps=1000000 \
  --report-steps=100 \
  --timeout 1.8 \
  --data-dir="${OUTPUT_DIR}"
```

## Output Structure

For each processed protein, the following directory structure is created:

```
data/
└── <pdb_id>/
    ├── raw/                      # Original PDB file
    │   └── <pdb_id>.pdb
    ├── processed/                # Prepared system
    │   ├── <pdb_id>_processed.pdb
    │   ├── <pdb_id>_processed_ligands_smiles.json
    │   ├── <pdb_id>_processed_ligands_cache.json
    │   ├── forcefield.json
    │   ├── <pdb_id>_process.log
    │   └── finished.txt
    ├── simulation/               # Simulation files
    │   ├── system.xml
    │   ├── integrator.xml
    │   ├── checkpoint.chk
    │   ├── log.txt
    │   ├── <pdb_id>_simulation.log
    │   ├── final_state.xml
    │   ├── final_state.pdb
    │   └── finished.txt
    └── result/                   # Trajectory data
        └── output_<pdb_id>.h5    # HDF5 trajectory file
```

### HDF5 Output Format

The HDF5 trajectory files contain:

- `coordinates`: Atomic positions (nm)
- `forces`: Atomic forces (kJ/mol/nm)
- `cell_lengths`: Periodic box dimensions (nm)
- `cell_angles`: Periodic box angles (degrees)
- `time`: Simulation time (ps)
- `potentialEnergy`: Potential energy (kJ/mol)
- `kineticEnergy`: Kinetic energy (kJ/mol)
- `topology`: System topology in JSON format

## Module Structure

### `module/function.py`
Utility functions for file management and topology operations.

### `module/preprocess.py`
Protein preparation pipeline:
- Download PDB files from RCSB
- Fix missing residues and atoms
- Add hydrogens at specified pH
- Add explicit or implicit solvent
- Parameterize ligands

### `module/simulation.py`
Core simulation engine:
- System setup and minimization
- Equilibration
- Production MD
- Checkpoint management
- Resumable simulations

### `module/ligands.py`
Ligand handling:
- RCSB ligand template retrieval
- SMILES-based parameterization
- GAFF force field generation
- Topology insertion

### `module/reporters.py`
Custom HDF5 reporter with:
- Threaded I/O for performance
- Force trajectory storage
- H5MD-compatible format
- Compression support

### `module/pdb_lookup.py`
RCSB database query utilities for ligand SMILES strings.

## Force Field Configuration

Default force fields:
- **Protein**: AMBER14
- **Water**: TIP3P-FB
- **Ligands**: GAFF (via OpenFF)
- **Implicit Solvent**: GBN2

Force field selection is automatic based on preparation options and stored in `forcefield.json`.

## Data Processing Workflow

### Cecilia Data Cleaning

The `cecilia_data_cleaning/` directory contains tools for processing existing MD datasets:

**Fix Tyrosine Residues:**
```bash
python cecilia_data_cleaning/fix_cecilia_tyrosol.py
```

**Generate Starting Poses via TICA:**
```bash
python cecilia_data_cleaning/tica_starting_pos/main.py
```

This performs time-lagged independent component analysis (TICA) on bond distances to identify diverse starting conformations for enhanced sampling.

## Performance Considerations

### GPU Utilization

- Each simulation requires ~2-4GB GPU memory
- Pool size should be a multiple of GPU count for efficiency
- Recommended: 2 simulations per GPU (pool_size = 2 × num_gpus)

### CPU Threading

- Set `OMP_NUM_THREADS` to match allocated cores
- Ligand parameterization temporarily limits threads to 2
- Typical: 32 threads for 4-GPU nodes

### I/O Performance

- HDF5 reporter uses threaded writes to minimize overhead
- `report_steps` controls output frequency and file size
- Compression can be enabled via `use_gzip=True`

## Troubleshooting

### Common Issues

**CUDA out of memory:**
- Reduce pool size
- Decrease number of atoms (remove water with `atomSubset`)

**Ligand parameterization fails:**
- Check RCSB connectivity
- Verify ligand is in PDB Chemical Component Dictionary
- Use `--remove-ligands` if parameterization not needed

**Simulation doesn't resume:**
- Ensure checkpoint files exist
- Check that HDF5 file isn't corrupted
- Verify simulation time matches last recorded frame

**Force field errors:**
- Check for non-standard residues
- Verify all ligands have templates
- Review preparation logs

## Contributing

We welcome contributions! Please use [GitHub Issues](../../issues) to:
- Report bugs
- Request features
- Ask questions
- Suggest improvements

## License

*License information will be added soon.*

## Acknowledgments

This work was performed on:
- Perlmutter at NERSC (National Energy Research Scientific Computing Center)
- Delta at NCSA (National Center for Supercomputing Applications)

Development supported by computational resources and expertise from these facilities.

## Contact

For questions and support, please open an issue on GitHub.

## References

- [OpenMM Documentation](http://docs.openmm.org/)
- [OpenFF Toolkit](https://open-forcefield-toolkit.readthedocs.io/)
- [MDTraj](https://mdtraj.org/)
- [H5MD Specification](https://www.nongnu.org/h5md/)
