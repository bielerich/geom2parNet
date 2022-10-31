#!/usr/bin/python3
import sys
sys.path.insert(0, "../lib")
import PostProcess as pp

def main():

    string ="run postProcess.py"
    print(string)

    PP = pp.PostProcess()

    PP.evaluateData()
    PP.plotData()

if __name__ == "__main__":
    main()
