"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os


class MyProject(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="gpu",
            help="Specify the partition to submit to."
        )


class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="gpuq",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="batch," "v100",
            help="Specify the partition to submit to."
        )

# Definition of project-related labels (classification)
@MyProject.label
def sampled(job):
    return job.doc.get("done")


@MyProject.label
def initialized(job):
    pass

@MyProject.label
def initial_run_done(job):
    return job.isfile("trajectory.gsd")

@MyProject.label
def equilibrated(job):
    return job.doc.equilibrated

@directives(executable="python -u")
@directives(ngpu=1)
@MyProject.operation
@MyProject.post(initial_run_done)
def sample(job):
    import hoomd_polymers
    from hoomd_polymers.systems import Pack
    import hoomd_polymers.forcefields
    from hoomd_polymers.sim import Simulation
    import mbuild as mb
    import foyer

    with job:
        print("JOB ID NUMBER:")
        print(job.id)
        system_file = "/home/madilynpaul/repos/forcefields/typed_mol2/perylene.mol2"
        mol_path = os.path.join(os.getcwd(), system_file)
        ff_file = "/home/madilynpaul/repos/forcefields/xml_files/perylene.xml"
        ff_path = os.path.join(os.getcwd(), ff_file)

        def espaloma_mol(file_path):
            mol = mb.load(file_path)
            for p in mol.particles():
                p.name = f"_{p.name}"
            return mol

        system = Pack(
                molecule=espaloma_mol,
                density=job.sp.density,
                n_mols=job.sp.n_compounds,
                mol_kwargs = {
                    "file_path": mol_path,
                },
                packing_expand_factor=5
        )

        system_ff = foyer.Forcefield(forcefield_files=ff_path)
        system.apply_forcefield(
                forcefield=system_ff,
                make_charge_neutral=True,
                remove_hydrogens=job.sp.remove_hydrogens,
                remove_charges=job.sp.remove_charges
        )

        job.doc.ref_distance = system.reference_distance
        job.doc.ref_mass = system.reference_mass
        job.doc.ref_energy = system.reference_energy

        gsd_path = os.path.join(job.ws, "trajectory.gsd")
        log_path = os.path.join(job.ws, "sim_data.txt")

        system_sim = Simulation(
            initial_state=system.hoomd_snapshot,
            forcefield=system.hoomd_forcefield,
            gsd_write_freq=job.sp.n_steps/1000,
            gsd_file_name=gsd_path,
            log_file_name=log_path,
            dt=job.sp.dt,
            log_write_freq=job.sp.n_steps/7000
        )
        target_box = system.target_box*10/job.doc.ref_distance
        job.doc.target_box = target_box

        system_sim.run_update_volume(
                final_box_lengths=target_box,
                n_steps=job.sp.shrink_steps,
                period=job.sp.shrink_period,
                tau_kt=job.sp.tau_kt,
                kT=job.sp.shrink_kT
        )
        system_sim.run_NVT(
                kT=job.sp.kT,
                n_steps=job.sp.n_steps,
                tau_kt=job.sp.tau_kt
        )

@directives(executable="python -u")
@directives(ngpu=1)
@MyProject.operation
@MyProject.pre(initial_run_done)
@MyProject.post(equilibrated)
def resume(job):
    import pickle

    import gsd.hoomd
    import hoomd_polymers
    from hoomd_polymers.systems import Pack
    import hoomd_polymers.forcefields
    from hoomd_polymers.sim import Simulation

    with job:
        print("JOB ID NUMBER:")
        print(job.id)
        print("Resuming Simulation")
        # Grab the correct pickle file for the forces
        if job.sp.remove_hydrogens and job.sp.remove_charges:
            ff_file = "/home/madilynpaul/notebooks/hoomdpolymers/perylene_study/esp/5.97kT/pickle_ff/forcefield_UA_unCH.pickle"
        with open(ff_file, "rb") as f:
            hoomd_ff = pickle.load(f)
    
        gsd_path = job.fn(f"trajectory{job.doc.run + 1}.gsd")
        log_path = job.fn(f"sim_data{job.doc.run + 1}.txt")
        if job.isfile("restart.gsd"):
            init_state = job.fn("restart.gsd")
        else:
            with gsd.hoomd.open(job.fn("trajectory.gsd")) as traj:
                init_state = traj[-1]

        system_sim = Simulation(
            initial_state=init_state,
            forcefield=hoomd_ff,
            gsd_write_freq=job.sp.n_steps/1000,
            gsd_file_name=gsd_path,
            log_file_name=log_path,
            dt=job.sp.dt,
            log_write_freq=job.sp.n_steps/7000
        )   

        system_sim.run_NVT(
                kT=job.sp.kT,
                n_steps=1e6,
                tau_kt=job.sp.tau_kt
        )   
        system_sim.save_restart_gsd(job.fn("restart.gsd"))
        job.doc.run += 1

@directives(executable="python -u")
@MyProject.operation
@MyProject.pre(equilibrated)
def coarse_grain(job):
    import grits
    from grits import CG_System, CG_Compound
    import gsd.hoomd
    import ele

    with gsd.hoomd.open(job.fn("cg_traj.gsd"),"wb") as new_traj:
        with gsd.hoomd.open(job.fn("trajectory.gsd")) as first_traj:
            new_traj.append(first_traj[0])
        if job.doc.run > 0:                
            with gsd.hoomd.open(job.fn(f"trajectory{job.doc.run}.gsd")) as last_traj:
                numframes = len(last_traj)
                print(numframes)
                for frame in last_traj[::50]:
                    new_traj.append(frame)
        if job.doc.run == 0:
            with gsd.hoomf.open(job.fn('trajectory.gsd')) as last_traj:
                for frame in last_traj[::50]:
                    new_traj.append(frame)

    perylene_dict = {
        "C0": ele.element_from_symbol("C")}
    cg_system = CG_System(
        gsdfile=job.fn("cg_traj.gsd"),
        beads={"_A":"c1cc2cccc3c4cccc5cccc(c(c1)c23)c45"},
        allow_overlap=True,
        add_hydrogens=True,
        conversion_dict=perylene_dict)
    cg_system.save(job.fn("perylene-cg.gsd"))
    cg_system.save_mapping(job.fn("perylene-cg-mapping.json"))

if __name__ == "__main__":
    MyProject(environment=Fry).main()
