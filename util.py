import numpy as np
import json

def create2DArray(start_val, end_val):
    chess_2d_array = []
    val = np.arange(start_val,end_val)
    for a in val:
        for b in val:
            chess_2d_array.append([a,b])
    return chess_2d_array

#corners: quello che ti dà la findChessboardCorners
def saveToJSON(dict, camera_number):
    with open(f"{camera_number}Fcorners.json", "a+") as outfile:
        json.dump(dict, outfile)
        return True
    return False