#!/usr/bin/env python

from potentiostat import Potentiostat
import sys
from io import StringIO
import numpy as np

if __name__ == "__main__":
    _, v_min, v_max, cycles, mV_s, step_hz, start_V, last_V, p_name, filename= sys.argv
    POTEN = Potentiostat(serial_port=p_name,baudrate=115200,device_ID = 2)
    POTEN.connect()

    try:
        start_V= float(start_V)
    except:
        start_V=None
    
    try:
        last_V= float(start_V)
    except:
        last_V=None

    rtn = POTEN.perform_CV(
        float(v_min)
        , float(v_max)
        , int(cycles)
        , float(mV_s)
        , int(step_hz)
        , start_V
        , last_V
    )

    np.savetxt(X=rtn, delimiter=',', fmt='%.3E,%.3E,%.3E,%d,%d,%.3E', fname = filename)

    POTEN.write_switch(False)
    POTEN.disconnect()
