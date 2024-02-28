import pandas as pd
import matplotlib.pyplot as plt

def preprocessDiscreteSpectrum(dataPath):

    #Initializing the dataframe of trajectory sample
    df = pd.read_csv(dataPath, names = ["X1", "X2"], header=None)

    #Initializing the parameters dictionary
    params = {}

    #Settings related to dataset
    params['lenTime'] = 51
    params['deltaT'] = 0.02

    #Settings related to loss function
    params['numShifts'] = 30
    params['reconLam'] = .1
    params['LinfLam'] = 10 ** (-7)
    params['L1Lam'] = 0.0
    params['L2Lam'] = 10 ** (-15)

    #Settings related to Network Architecture
    params['inputSize'] = 2
    params['hiddenSize'] = 30
    params['hiddenLayer'] = 2
    params['latentSize'] = 2

    return df, params


if __name__ == "__main__":
    df, params = preprocessDiscreteSpectrum("data/DiscreteSpectrumExample_train1_x.csv")
    plt.scatter(df["X1"], df["X2"])
    plt.show()
    print(params)
