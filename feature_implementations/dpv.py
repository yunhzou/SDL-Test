#!/usr/bin/env python

from potentiostat import Potentiostat
import sys
from io import StringIO
import numpy as np

if __name__ == "__main__":
    _, min_V, pulse_V, step_V, max_V, potential_hold_ms, pulse_hold_ms, voltage_hold_s, serial_port, poten_id, ref, filename = sys.argv
    
    POTEN = Potentiostat(serial_port)
    POTEN.connect()


    rtn = POTEN.perform_CDPV(
        min_V=float(min_V)
        , pulse_V=float(pulse_V)
        , step_V=float(step_V)
        , max_V=float(max_V)
        , potential_hold_ms=float(potential_hold_ms)
        , pulse_hold_ms=float(pulse_hold_ms)
        , voltage_hold_s=float(voltage_hold_s)
    )


    np.savetxt(X=rtn, delimiter=',', fmt='%.3E,%.3E,%.3E,%d,%d,%.3E', fname = filename)

    POTEN.write_switch(False)
    POTEN.disconnect()
