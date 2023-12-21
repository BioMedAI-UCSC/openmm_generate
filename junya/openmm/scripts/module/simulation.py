from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np
import h5py
import os
from module import ligands
from module import function
from module.reporters import ExtendedH5MDReporter

def run(pdbid=str, input_pdb_path=str, steps=100, load_ligand_smiles=True, atomSubset=None):
    """
    Run the simulation for the given PDB ID.

    Args:
        pdbid (str): The PDB ID.
        input_pdb_path (str): The path to the input PDB file.
        atomSubset (list or None): List of atom indices to subset. Defaults to None.

    Returns:
        None
    """
    
    
    print(f"Start simulation of {pdbid}...")
    
    # Input Files
    pdb = PDBFile(input_pdb_path)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    if load_ligand_smiles:
        input_ligands_path = os.path.splitext(input_pdb_path)[0]+"_ligands_smiles.json"
        template_cache_path = os.path.splitext(input_pdb_path)[0]+"_ligands_cache.json"
        if os.path.exists(input_ligands_path):
            ligands.add_ff_template_generator_from_json(forcefield, input_ligands_path, template_cache_path)
        else:
            print(f"'{input_ligands_path}' does not exist, skipping template generation.")

    # System Configuration
    nonbondedMethod = PME
    nonbondedCutoff = 1.0*nanometers
    ewaldErrorTolerance = 0.0005
    constraints = HBonds
    rigidWater = True
    constraintTolerance = 0.000001
    hydrogenMass = 1.5*amu

    # Integration Options
    dt = 0.002*picoseconds
    temperature = 300*kelvin
    friction = 1.0/picosecond
    pressure = 1.0*atmospheres
    barostatInterval = 25

    # Simulation Options
    equilibrationSteps = 10
    reportInterval = min(int(steps/10), 1000)
    platformNames = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    if 'CUDA' in platformNames:
        platform = Platform.getPlatformByName('CUDA')
        platformProperties = {'Precision': 'single'}
    elif 'OpenCL'in platformNames:
        platform = Platform.getPlatformByName('OpenCL')
        platformProperties = {'Precision': 'single'}
    else:
        platform = None
        platformProperties = {}
    print(f"Simulation platform: {platform.getName()}, {platformProperties}")
    
    # Reporters
    try:
        hdf5Reporter = None

        hdf5Reporter = ExtendedH5MDReporter(function.get_data_path(f'{pdbid}/result/output_{pdbid}.h5'), 1, total_steps=steps, atom_subset=atomSubset)
        dataReporter = StateDataReporter(function.get_data_path(f'{pdbid}/simulation/log.txt'), reportInterval, totalSteps=steps,
            step=True, speed=True, progress=True, potentialEnergy=True, temperature=True, separator='\t')
        checkpointReporter = CheckpointReporter(function.get_data_path(f'{pdbid}/simulation/checkpoint.chk'), reportInterval)

        # Prepare the Simulation
        print('Building system...')
        topology = pdb.topology
        positions = pdb.positions
        system = forcefield.createSystem(topology, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
            constraints=constraints, rigidWater=rigidWater, ewaldErrorTolerance=ewaldErrorTolerance, hydrogenMass=hydrogenMass)
        system.addForce(MonteCarloBarostat(pressure, temperature, barostatInterval))
        integrator = LangevinMiddleIntegrator(temperature, friction, dt)
        integrator.setConstraintTolerance(constraintTolerance)
        simulation = Simulation(topology, system, integrator, platform, platformProperties)
        simulation.context.setPositions(positions)

        # Write XML serialized objects
        with open(function.get_data_path(f"{pdbid}/simulation/system.xml"), mode="w") as file:
            file.write(XmlSerializer.serialize(system))
        with open(function.get_data_path(f"{pdbid}/simulation/integrator.xml"), mode="w") as file:
            file.write(XmlSerializer.serialize(integrator))

        # Minimize and Equilibrate
        print('Performing energy minimization...')
        simulation.minimizeEnergy()
        print('Equilibrating...')
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(equilibrationSteps)

        # Simulate
        print('Simulating...')
        simulation.reporters.append(hdf5Reporter)
        simulation.reporters.append(dataReporter)
        simulation.reporters.append(checkpointReporter)
        simulation.currentStep = 0

        from sys import stdout
        simulation.reporters.append(StateDataReporter(stdout, reportInterval, step=True,
            progress=True, remainingTime=True, speed=True, totalSteps=steps,
            separator="\t"))

        # Propaggerate the simulation and save the data
        simulation.step(steps)

    finally:
        # close the reporters
        if hdf5Reporter:
            hdf5Reporter.close()

    # Write file with final simulation state
    simulation.saveState(function.get_data_path(f"{pdbid}/simulation/final_state.xml"))
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=system.usesPeriodicBoundaryConditions())
    with open(function.get_data_path(f"{pdbid}/simulation/final_state.pdb"), mode="w") as file:
        PDBFile.writeFile(simulation.topology, state.getPositions(), file)

    # Assert the data
    with h5py.File(function.get_data_path(f"{pdbid}/result/output_{pdbid}.h5"), "r") as f:
        # Check the shape of the data
        assert f["coordinates"].shape == f["forces"].shape
        # Check the dimension of the data
        for key in ["coordinates", "forces"]:
            print(key, f[key].shape)
            assert f[key].shape[0] == steps
            assert f[key].shape[1] == len(atomSubset)
            assert f[key].shape[2] == 3
            # Check if the data is not the same
            for i in range(10-1):
                assert f[key][0,0,0] != f[key][i+1,0,0]

    print(f"Simulation of {pdbid} is done.")
    print(f"Result is here: {function.get_data_path(f'{pdbid}/result/output_{pdbid}.h5')}\n")
