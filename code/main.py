from datapr import getDataFromCSV
from sklearn.linear_model import LinearRegression, _base
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, mean_squared_error
from sklearn.ensemble import IsolationForest
from typing import Any
import numpy as np

class ModelGRE :
    def __init__(self, url : str, **kwargs : Any) -> None:

        self.__data__ = getDataFromCSV(url, **kwargs)
        self.__X_train__, self.__X_test__, self.__Y_train__, self.__Y_test__ = self.__trainTestSplitData__(0.3)

        self.__model__ = LinearRegression()

    def __XLabelsWithoutEmissions__(self, X : np.ndarray, Y : np.ndarray) :

        return IsolationForest().fit_predict(X, Y)

    def __XYWithoutEmissions__(self, X : np.ndarray, Y : np.ndarray) :

        XX = self.__XLabelsWithoutEmissions__(X, Y)

        X, Y = X[XX != -1], Y[XX != -1]

        return X, Y

    def __trainTestSplitData__(self, test_size : int) :

        Y = self.__data__['Chance of Admit']
        X = self.__data__.iloc[:, :-1]

        X, Y = self.__XYWithoutEmissions__(X, Y)

        return train_test_split(X, Y, test_size=test_size)

    def __fit__(self) :

        self.__model__.fit(self.__X_train__, self.__Y_train__)

    def __predict__(self) :

        return self.__model__.predict(self.__X_test__)

    @property
    def Y_pred(self) :

        self.__fit__()

        return self.__predict__()

    def MSE(self) :

        return mean_squared_error(self.__Y_test__, self.Y_pred)


if __name__ == '__main__' :

    modelGRE = ModelGRE("data/AdmissionPredict.csv")

    print(f'MSE : {modelGRE.MSE()}')

