#!/usr/bin/python3
import sys
sys.path.insert(0, "../lib")
import NeuralNetwork as nn

def main():

    string ="run trainNetwork.py"
    print(string)

    NN = nn.NeuralNetwork()

    NN.trainNetwork()


if __name__ == "__main__":
    main()
