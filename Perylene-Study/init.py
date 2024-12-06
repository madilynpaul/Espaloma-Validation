#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories.
The result of running this file is the creation of a signac workspace:
    - signac.rc file containing the project name
    - signac_statepoints.json summary for the entire workspace
    - workspace/ directory that contains a sub-directory of every individual statepoint
    - signac_statepoints.json within each individual statepoint sub-directory.

"""

import signac
import logging
from collections import OrderedDict
from itertools import product


def get_parameters():
    '''
    '''
    parameters = OrderedDict()

    ### SYSTEM GENERATION PARAMETERS ###
    parameters["density"] = [0.5,1.0,1.5]
    parameters["n_compounds"] = [250]
    parameters["system_kwargs"] = [None]
    parameters["remove_hydrogens"] = [True]
    parameters["remove_charges"] = [True]


    ### SIMULATION PARAMETERS ###
    parameters["tau_kt"] = [0.01]
    parameters["dt"] = [0.0001]
    parameters["r_cut"] = [2.5]
    parameters["sim_seed"] = [42]
    parameters["shrink_steps"] = [5e6]
    parameters["shrink_period"] = [10000]
    parameters["shrink_kT"] = [8.0]
    ### Quench related parameters ###
    parameters["kT"] = [1.15,5.749,11.5,17.25]
    parameters["n_steps"] = [1e6]
    return list(parameters.keys()), list(product(*parameters.values()))

def main():
    project = signac.init_project("perylene") # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create the generate jobs
    for params in param_combinations:
        parent_statepoint = dict(zip(param_names, params))
        parent_job = project.open_job(parent_statepoint)
        parent_job.init()
        parent_job.doc.setdefault("done", False)
        parent_job.doc.setdefault("equilibrated", False)
        parent_job.doc.setdefault("run", 0)

    project.write_statepoints()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
