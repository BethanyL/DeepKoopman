# DeepKoopman
neural networks to learn Koopman eigenfunctions

Code for the paper ["Deep learning for universal linear embeddings of nonlinear dynamics"](https://www.nature.com/articles/s41467-018-07210-0) by Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton

To run code:

1. Clone respository.
2. In the data directory, recreate desired dataset(s) by running DiscreteSpectrumExample, Pendulum, FluidFlowOnAttractor, and/or FluidFlowBox in Matlab. (or email to ask for the datasets)
3. Back in the main directory, run desired experiment(s) with python.

Notes on running the Python experiments:
- A GPU is recommended but not required. The code can be run on a GPU or CPU without any changes.
- The paper contains results on the four datasets. These were the best results from running scripts that do a random parameter search (DiscreteSpectrumExampleExperiment.py, PendulumExperiment.py, FluidFlowOnAttractorExperiment.py, and FluidFlowBoxExperiment.py). 
- To train networks using the specific parameters that produced the results in the paper instead of doing a parameter search, run DiscreteSpectrumExampleExperimentBestParams.py, PendulumExperimentBestParams.py, FluidFlowOnAttractorExperimentBestParams.py, and FluidFlowBoxExperimentBestParams.py.
- The experiment scripts include a loop over 200 random experiments (random parameters and random initializations of weights). You'll probably want to kill off the script earlier than that!
- Each random experiment can run up to params['max_time'] (in these experiments, 4 or 6 hours) but may be automatically terminated earlier if the error is not decreasing enough. If one experiment is not doing well, the script moves on to another random experiment.
- If the code decides to end an experiment, it saves the current results. It also saves every hour. 
