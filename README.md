# DeepKoopman
neural networks to learn Koopman eigenfunctions

Code for the paper ["Deep learning for universal linear embeddings of nonlinear dynamics"](https://www.nature.com/articles/s41467-018-07210-0) by Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton

To run code:

1. Clone respository.
2. In the data directory, recreate desired dataset(s) by running DiscreteSpectrumExample, Pendulum, FluidFlowOnAttractor, and/or FluidFlowBox in Matlab (or download the datasets from Box [here](https://anl.box.com/s/9s29juzu892dfkhgxa1n1q4mj63nxabn)).
3. Back in the main directory, run desired experiment(s) with python.

Notes on running the Python experiments:
- A GPU is recommended but not required. The code can be run on a GPU or CPU without any changes.
- The paper contains results on the four datasets. These were the best results from running scripts that do a random parameter search (DiscreteSpectrumExampleExperiment.py, PendulumExperiment.py, FluidFlowOnAttractorExperiment.py, and FluidFlowBoxExperiment.py). 
- To train networks using the specific parameters that produced the results in the paper instead of doing a parameter search, run DiscreteSpectrumExampleExperimentBestParams.py, PendulumExperimentBestParams.py, FluidFlowOnAttractorExperimentBestParams.py, and FluidFlowBoxExperimentBestParams.py.
- The experiment scripts include a loop over 200 random experiments (random parameters and random initializations of weights). You'll probably want to kill off the script earlier than that!
- Each random experiment can run up to params['max_time'] (in these experiments, 4 or 6 hours) but may be automatically terminated earlier if the error is not decreasing enough. If one experiment is not doing well, the script moves on to another random experiment.
- If the code decides to end an experiment, it saves the current results. It also saves every hour. 

Postprocessing:
- You might want to use something like ./postprocessing/InvestigateResultsExample.ipynb to check out your results. Which of your models has the best validation error so far? How does validation error compare to your hyperparameter choices? 
- To see what I did to dive into a particular trained deep learning model on a dataset, see the notebooks ./postprocessing/BestModel-DiscreteSpectrumExample.ipynb, ./postprocessing/BestModel-Pendulum.ipynb, etc. These notebooks also show how I calculated numbers and created figures for the paper. 

New to deep learning? Here is some context:
- It is currently normal in deep learning to need to try a range of hyperparameters ("hyperparameter search"). For example: how many layers should your network have? How wide should each layer be? You try some options and pick the best result. (See next bullet point.) Further, the random initialization of your weights matters, so (unless you fix the seed of your random number generator) even with fixed hyperparameters, you can re-run your training multiple times and get different models with different errors. I didn't fix my seeds, so if you re-run my code multiple times, you can get different models and errors. 
- It is standard to split your data into three sets: training, validation, and testing. You fit your neural network model to your training data. You only use the validation data to compare different models and choose the best one. The error on your validation data estimates how well your model will generalize to new data. You set aside the testing data even further. You only calcuate the error on the test data at the very end, after you've commited to a particular model. This should give a better estimate of how well your model will generalize, since you may have already heavily relied on your validation data when choosing a model. 

## Citation
```
@article{lusch2018deep,
  title={Deep learning for universal linear embeddings of nonlinear dynamics},
  author={Lusch, Bethany and Kutz, J Nathan and Brunton, Steven L},
  journal={Nature Communications},
  volume={9},
  number={1},
  pages={4950},
  year={2018},
  publisher={Nature Publishing Group},
  Doi = {10.1038/s41467-018-07210-0}
}
```
