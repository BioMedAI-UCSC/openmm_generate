from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np
from mdtraj.reporters import HDF5Reporter
import h5py
from mdtraj.reporters import HDF5Reporter


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

def update_numpyfile(file_p=str, file_f=str, positions=np.array, forces=np.array):
    """
    Update the existing numpy files with new positions and forces.

    Parameters:
    file_p (str): Path to the numpy file containing existing positions.
    file_f (str): Path to the numpy file containing existing forces.
    positions (np.array): Array of new positions to be added.
    forces (np.array): Array of new forces to be added.

    Returns:
    None
    """
    # Load existing data
    existing_positions = np.load(file_p)
    existing_forces = np.load(file_f)
    # Concatenate the data
    positions_to_save = np.concatenate([existing_positions, positions])
    forces_to_save = np.concatenate([existing_forces, forces])
    # Save the data
    np.save(file_p, positions_to_save)
    np.save(file_f, forces_to_save)
            

def run(pdbid=str, input_pdb_path=str, atomSubset=None):
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
    steps = 100
    equilibrationSteps = 10
    reportInterval = int(steps/10)
    platform = Platform.getPlatformByName('CUDA')
    platformProperties = {'Precision': 'single'}

    # Reporters
    hdf5Reporter = HDF5Reporter(f'../data/{pdbid}/result/output_{pdbid}.h5', reportInterval=1, atomSubset=atomSubset)
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
    
    file_p = f"../data/{pdbid}/simulation/positions.npy"
    file_f = f"../data/{pdbid}/simulation/forces.npy"
    # initialize the numpyfiles
    np.save(file_p, np.empty((0, 3)))
    np.save(file_f, np.empty((0, 3)))
    # Get postions and forces at each frame
    for _ in range(steps):
        simulation.step(1)
        # get positions and forces
        positions, forces = get_pos_force(simulation, atomSubset)
        # save the data to numpyfiles
        update_numpyfile(file_p, file_f, positions, forces)
    
    # close the reporters
    hdf5Reporter.close()


    # Write file with final simulation state
    simulation.saveState(f"../data/{pdbid}/simulation/final_state.xml")
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=system.usesPeriodicBoundaryConditions())
    with open(f"../data/{pdbid}/simulation/final_state.pdbx", mode="w") as file:
        PDBxFile.writeFile(simulation.topology, state.getPositions(), file)

    
    # Save positions and forces to HDF5 file
    # Load the data
    forces = np.load(f"../data/{pdbid}/simulation/forces.npy")
    positions = np.load(f"../data/{pdbid}/simulation/positions.npy")
    # Split the data
    forces = np.array(np.split(forces, steps))
    positions = np.array(np.split(positions, steps))
    # save the data
    with h5py.File(f"../data/{pdbid}/result/output_{pdbid}.h5", "a") as f:
        if "forces" not in f.keys():
            f.create_dataset("forces", data=forces)
        if "positions" not in f.keys():
            f.create_dataset("positions", data=positions)
    del forces, positions
    
    # delete the numpyfiles
    # for file_path in [file_p, file_f]:
    #     try:
    #         os.remove(file_path)
    #     except OSError as e:
    #         print(f"Error: {e.filename} - {e.strerror}")        
    
    # Assert the data
    with h5py.File(f"../data/{pdbid}/result/output_{pdbid}.h5", "a") as f:
        # Check the shape of the data
        assert f["positions"].shape == f["forces"].shape
        # Check the dimension of the data
        for key in ["positions", "forces"]:
            assert f[key].shape[0] == steps
            assert f[key].shape[1] == len(atomSubset)
            assert f[key].shape[2] == 3
            # Check if the data is not the same
            for i in range(10-1):
                assert f[key][0,0,0] != f[key][i+1,0,0]

        # # Check if the data is the same
        for i in range(5):
            for j in range(5):
                assert f["coordinates"][i,j,0] == f["positions"][i,j,0], "The coordinates and positions are not the same."
        
        # FIXME : The y values of coordinates(from HDF5Reporter) and those of positions(from get_pos_force) dont match. Why?
        # x and z values match, but only y values don't.
        # I checked the initial positions and found the coordinates were not correct and the positions were correct.
        # So, I think the problem is in the HDF5Reporter.

        
    print(f"Simulation of {pdbid} is done.")
    print(f"Result is here: {f'../data/{pdbid}/result/output_{pdbid}.h5'}")
