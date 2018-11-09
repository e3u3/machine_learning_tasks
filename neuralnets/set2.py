import numpy as np
from load_mnist import load_dataset


def main():
    
    trainx, trainy, testx, testy, valx, valy = load_dataset()
    print(np.shape(trainx))


if __name__ == 'main':
   main()

 
