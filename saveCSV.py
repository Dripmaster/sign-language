import pandas as pd

def csvSave(x,name):
    df = pd.DataFrame(x)
    df.head()
    r = df.to_csv(name)
    return r

def csvLoad(name):
    df = pd.read_csv(name,index_col=0)
    return df