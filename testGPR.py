from gpcam import AutonomousExperimenterGP
import numpy as np

def instrument(data):
    for entry in data:
        print("I want to know the y_data at: ", entry["x_data"])
        entry["y_data"] = np.sin(np.linalg.norm(entry["x_data"]))
        print("I received ",entry["y_data"])
        print("")
    return data

##set up your parameter space
parameters = np.array([[3.0,45.8],
                       [4.0,47.0]])

##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
init_hyperparameters = np.array([1,1,1])
hyperparameter_bounds =  np.array([[0.01,100],[0.01,100.0],[0.01,100]])

##let's initialize the autonomous experimenter ...
my_ae = AutonomousExperimenterGP(parameters, init_hyperparameters,
                                 hyperparameter_bounds,instrument_function = instrument,
                                 init_dataset_size=10, info=False)
#...train...
my_ae.train()

#...and run. That's it. You successfully executed an autonomous experiment.
my_ae.go(N = 100)