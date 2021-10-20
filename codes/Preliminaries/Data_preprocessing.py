from numpy.lib.npyio import load
import torch
import numpy as np
import pandas as pd
import os

def load_data():
    if not os.path.exists('.house_tiny.csv'):
        with open('.house_tiny.csv', 'w') as f:
            f.write('NumRooms, Alley, Price\n')
            f.write('NA, Lingshui, 12000\n')
            f.write('4, Gaoxin, 20000\n')
            f.write('3, Ganjingzi, 8000\n')
            f.write('2, NA, 10000\n')
            f.write('NA, NA, 1000\n')
        with open('.house_tiny.csv', 'r') as f:
            houses = pd.read_csv(f)
    else:
        try:
            with open('.house_tiny.csv', 'r') as f:
                houses = pd.read_csv(f)
        except:
            print("open file failed")
    
    return houses


def data_preprocess():
    houses = load_data()
    print(houses.head(), houses.shape)
    input, output = houses.iloc[:, :2], houses.iloc[:, 2]
    print(input)
    print(output)
    print(houses.iloc[4, 1])
    input = input.fillna(input.mean())
    print(input)
    input = pd.get_dummies(input, dummy_na=False)
    print(input)
    print(input.values)
    print(output.values)
    input = np.array(input.values, dtype=np.float16)
    output = np.array(output.values, dtype=np.float16)
    input = torch.tensor(input)
    output = torch.tensor(output)
    print(input)
    print(output)

if __name__ == "__main__":
    data_preprocess()