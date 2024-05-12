import numpy as np
import json
import datetime

def create2DArray(start_val, end_val):
    chess_2d_array = []
    val = np.arange(start_val,end_val)
    for a in val:
        for b in val:
            chess_2d_array.append([a,b])
    return chess_2d_array

#corners: quello che ti dà la findChessboardCorners
def saveToJSON(dict: dict, camera_number: int):
    with open(f"{camera_number}Fcorners.json", "a+") as outfile:
        json.dump(dict, outfile)
        return True
    return False

def getSeconds(dat: datetime):
    t = dat.time()
    return (t.hour * 60 * 60 * 1000) + (t.minute * 60 * 1000) + (t.second * 1000) + t.microsecond 