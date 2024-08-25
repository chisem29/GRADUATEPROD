import pandas as pd
from typing import Any

def getDataFromCSV(dir : str, **kwargs : Any) :
    return pd.read_csv(dir, **kwargs)

