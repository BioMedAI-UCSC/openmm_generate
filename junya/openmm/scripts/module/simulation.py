from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np
from mdtraj.reporters import HDF5Reporter
import h5py
import datetime
from importlib import reload
from mdtraj.reporters import HDF5Reporter


class ForceReporter(object):
    def __init__(self, file, reportInterval, atomSubset=None):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self._atomSubset = atomSubset

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        f = state.getForces().value_in_unit(kilojoules/mole/nanometer)
        # Save only the atoms of protein
        if self._atomSubset is not None:
            forces = [f[i] for i in self._atomSubset]
        for f in forces:
            self._out.write('%g %g %g\n' % (f[0], f[1], f[2]))

class PositionReporter(object):
    def __init__(self, file, reportInterval, atomSubset=None):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self._atomSubset = atomSubset

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        p = state.getPositions(asNumpy=True).value_in_unit(angstrom)
        # Save only the atoms of protein
        if self._atomSubset is not None:
            positions = [p[i] for i in self._atomSubset]
        for p in positions:
            self._out.write('%g %g %g\n' % (p[0], p[1], p[2]))


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
    equilibrationSteps = 100
    reportInterval = int(steps/10)
    platform = Platform.getPlatformByName('CUDA')
    platformProperties = {'Precision': 'single'}

    # Reporters
    hdf5Reporter = HDF5Reporter(f'../data/{pdbid}/result/output_{pdbid}.h5', reportInterval, atomSubset=atomSubset)
    dataReporter = StateDataReporter(f'../data/{pdbid}/simulation/log.txt', reportInterval, totalSteps=steps,
        step=True, speed=True, progress=True, potentialEnergy=True, temperature=True, separator='\t')
    checkpointReporter = CheckpointReporter(f'../data/{pdbid}/simulation/checkpoint.chk', reportInterval)
    forcereporter = ForceReporter(f'../data/{pdbid}/simulation/force.txt', reportInterval=1, atomSubset=atomSubset)
    positionReporter = PositionReporter(f'../data/{pdbid}/simulation/position.txt', reportInterval=1, atomSubset=atomSubset)

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
    simulation.reporters.append(forcereporter)
    simulation.reporters.append(positionReporter)
    simulation.currentStep = 0

    from sys import stdout
    simulation.reporters.append(StateDataReporter(stdout, reportInterval, step=True, 
        progress=True, remainingTime=True, speed=True, totalSteps=steps, 
        separator="\t"))


    positions = []
    forces = []
    # Get postions and forces at each frame
    for _ in range(steps):
        simulation.step(1)
        # Create state object
        state = simulation.context.getState(getPositions=True, getForces=True)
        p = state.getPositions(asNumpy=True).value_in_unit(angstrom)
        f = state.getForces(asNumpy=True).value_in_unit(kilojoules/mole/nanometer)
        
        # Save only the atoms of protein
        if atomSubset is not None:
            p = [p[i] for i in atomSubset]
            f = [f[i] for i in atomSubset]
        positions.append(p)
        forces.append(f)
    positions = np.array(positions)
    forces = np.array(forces)
    
    # close hdf5 reporter
    hdf5Reporter.close()

    # Write file with final simulation state
    simulation.saveState(f"../data/{pdbid}/simulation/final_state.xml")

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=system.usesPeriodicBoundaryConditions())
    with open(f"../data/{pdbid}/simulation/final_state.pdbx", mode="w") as file:
        PDBxFile.writeFile(simulation.topology, state.getPositions(), file)

    
    # Write positions and forces to output h5 file
    with h5py.File(f"../data/{pdbid}/result/output_{pdbid}.h5", "a") as f:
        if "forces" not in f.keys():
            f.create_dataset("forces", data=forces)
        if "positions" not in f.keys():
            f.create_dataset("positions", data=positions)

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
            for i in range(100-1):
                assert f[key][0,0,0] != f[key][i+1,0,0]
    
    print(f"Simulation of {pdbid} is done.")
    print(f"Result is here: {f'../data/{pdbid}/result/output_{pdbid}.h5'}")
