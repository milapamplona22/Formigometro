Last update: 19.06.2021
# Optimizers
Process optimization. Currently uses Bayesian optimization. Parallelizes execution between samples under the same set of parameters.


- input = .csv table:
    +   1st column: video, or input of the command to be executed
    +   2nd column: respective label data (ground truth)

In the current case, we pair the output of the detector (../yolo/yoloV5Detect.py) and manual annotation from CVAT (.xml) of each video, and as parameters to be optimized, the RRANSAC parameters.

The specialization of the task should actually be performed outside the optimizer folder, as a specialization for each case. However, as an example model, the current case will be kept as an example (task.py).

*otmMngr.py*
The optimizer to be used.

*runTasks.py*
Executes a program as called from the command line, with the possibility of setting a timeout and being subject to interruptions.

*MOTmetrics.py*
Uses the MOTmetrics Python package to calculate tracking errors.
