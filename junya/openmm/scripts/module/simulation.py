from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np
import h5py
import os
from module import ligands
from module.reporters import ExtendedH5MDReporter

def get_pos_force(simulation=Simulation, atomSubset=None):
    """
    Get the positions and forces of the atoms in the simulation.

    Parameters:
    simulation (Simulation): The OpenMM simulation object.
    atomSubset (list): A list of atom indices to subset the positions and forces. Default is None.

    Returns:
    positions (numpy.ndarray): The positions of the atoms.
    forces (numpy.ndarray): The forces acting on the atoms.
    """
    state = simulation.context.getState(getPositions=True, getForces=True)
    p = state.getPositions(asNumpy=True).value_in_unit(nanometer)
    f = state.getForces(asNumpy=True).value_in_unit(kilojoules/mole/nanometer)
    
    # Save only the atoms of protein
    if atomSubset is not None:
        positions = p[atomSubset]
        forces = f[atomSubset]
    else:
        positions = np.array(p)
        forces = np.array(f)

    return positions, forces

def insert_or_create_h5(h5file, name, data, steps):
    """
    Append data to a dataset in the passed HDF5 file, creating the dataset if necessary.

    Args:
        h5file (h5py.File): The file to write to.
        name (str): The name of the dataset to append to.
        data (numpy.array): The data to append.
        steps (int): The maximum number of steps in the simulation, which will set the
                     maximum size for the dataset if it needs to be created.
    Returns:
        None
    """
    insertable_data = data.reshape(tuple([1] + list(data.shape)))
    if name not in h5file.keys():
        h5file.create_dataset(name, data=insertable_data, chunks=True, maxshape=tuple([steps] + list(data.shape)))
    else:
        h5file[name].resize((h5file[name].shape[0] + insertable_data.shape[0]), axis = 0)
        h5file[name][-insertable_data.shape[0]:] = insertable_data

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

        hdf5Reporter = ExtendedH5MDReporter(f'../data/{pdbid}/result/output_{pdbid}.h5', 1, total_steps=steps, atom_subset=atomSubset, use_gzip=True)
        dataReporter = StateDataReporter(f'../data/{pdbid}/simulation/log.txt', reportInterval, totalSteps=steps,
            step=True, speed=True, progress=True, potentialEnergy=True, temperature=True, separator='\t')
        checkpointReporter = CheckpointReporter(f'../data/{pdbid}/simulation/checkpoint.chk', reportInterval)

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
        with open(f"../data/{pdbid}/simulation/system.xml", mode="w") as file:
            file.write(XmlSerializer.serialize(system))
        with open(f"../data/{pdbid}/simulation/integrator.xml", mode="w") as file:
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
    simulation.saveState(f"../data/{pdbid}/simulation/final_state.xml")
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=system.usesPeriodicBoundaryConditions())
    with open(f"../data/{pdbid}/simulation/final_state.pdb", mode="w") as file:
        PDBFile.writeFile(simulation.topology, state.getPositions(), file)

    # Assert the data
    with h5py.File(f"../data/{pdbid}/result/output_{pdbid}.h5", "r") as f:
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

        # # # Check if the data is the same
        # for i in range(5):
        #     for j in range(5):
        #         assert f["coordinates"][i,j,0] == f["positions"][i,j,0], "The coordinates and positions are not the same."
        
        # FIXME : In 10GS, the y values of coordinates(from HDF5Reporter) and those of positions(from get_pos_force) dont match. Why?
        # x and z values match, but only y values don't.
        # I checked the initial positions and found the coordinates were not correct and the positions were correct.
        # So, I think the problem is in the HDF5Reporter.

        
    print(f"Simulation of {pdbid} is done.")
    print(f"Result is here: {f'../data/{pdbid}/result/output_{pdbid}.h5'}")
