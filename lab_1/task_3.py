import numpy as np
import pandas as pd
import time

# 5variant
# Задан Series объект s, узнать, есть ли значение 7 в объекте s.

def main():
    arr = [1,3,5,np.nan,6,8]
    s = pd.Series(arr)

    if s.isin([7]).any():
        print('yes, there is 7')
    else:
        print('no, there is not 7')
    

if __name__ == "__main__":
    main()
