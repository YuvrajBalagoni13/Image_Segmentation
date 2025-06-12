from Engine import train
from Utils import Plots

def main():
    print("Hello from Image Segmentation!")
    result = train.Train(epochs=5)
    Plots.PlotResultCurves(result)

if __name__ == "__main__":
    main()
