import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TrajectoryDataset(Dataset):
    def __init__(self, filePath):
        #Load the dataset
        fullDataset = pd.read_csv(filePath, header= None, names= ['X0', 'X1'])

        #Create initial values in the dataset
        self.X = fullDataset[fullDataset.index % 51 == 0]
        self.X.reset_index(drop= True, inplace= True)

        #Create the target from the dataset (i.e the full trajectory of 50 steps)
        self.y = pd.DataFrame(columns=['Trajectory'])

        for i in range(0, len(fullDataset), 51):
            rows = fullDataset.iloc[i:i+50]
            trajectory = [[row['X0'], row['X1']] for _, row in rows.iterrows()]
            self.y.loc[i // 50] = [trajectory]


    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return torch.tensor([self.X.iloc[index, 0], self.X.iloc[index, 1]]), [torch.tensor(item) for item in self.y.iloc[index, 0]]

def getDataLoader(dataset, batchSize=128):
    return DataLoader(dataset, batch_size= batchSize)



if __name__ == '__main__':

    testDataset = TrajectoryDataset('Koopman (Local)/data/DiscreteSpectrumExample_train1_x.csv') 
    print(testDataset[0])
    testDataloader = getDataLoader(testDataset)
    print(testDataloader)
