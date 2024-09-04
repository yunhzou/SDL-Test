#!/usr/bin/env python3

from typing import NamedTuple
from pathlib import Path
from .potentiostat import Potentiostat
import time
from ..generated.potenserver import (
    PotenServerBase,
    PrepareCompund_IntermediateResponses,
    PrepareCompund_Responses,
    Rinse_IntermediateResponses,
    Rinse_Responses,
    RunExp_IntermediateResponses,
    RunExp_Responses,
    RunReference_IntermediateResponses,
    RunReference_Responses,
)
import asyncio
from io import StringIO
import subprocess
import uuid
from uuid import uuid1
import sys
class Compound(NamedTuple):
    position: int
    volume: float

class PotCfg(NamedTuple):
    v_min: float
    v_max: float
    cycles: int
    steps: int


class CV_Cfg(NamedTuple):
    v_min: float
    v_max: float
    cycles: int
    mV_s: float
    step_hz: float
    start_V: float
    last_V: float

class DPV_Cfg(NamedTuple):
    min_V: float
    pulse_V: float
    step_V: float
    max_V: float
    potential_hold_ms: int
    pulse_hold_ms: int
    voltage_hold_s: float

class ExpCfg(NamedTuple):
    quantity_buffer:float
    quantity_electrolyte:float
    num_mixings: int
    ligand: Compound
    metal: Compound

class RunCfg(NamedTuple):
    # potentiostat: PotCfg
    experiment: ExpCfg
    CV: CV_Cfg
    DPV: DPV_Cfg

class RefCfg(NamedTuple):
    # potentiostat: PotCfg
    CV: CV_Cfg
    DPV: DPV_Cfg


def load_cfg_pot(
    dict_cfg: dict
) -> PotCfg:
    return PotCfg(**dict_cfg)

def load_CV_cfg(
    dict_cfg: dict
) -> CV_Cfg:
    return CV_Cfg(**dict_cfg)

def load_DPV_cfg(
    dict_cfg: dict
) -> CV_Cfg:
    return DPV_Cfg(**dict_cfg)


def load_cfg_exp(
    dict_cfg: dict
) -> ExpCfg:
    return ExpCfg (
        num_mixings = dict_cfg["num_mixings"]
        , quantity_buffer = dict_cfg["quantity_buffer"]
        , quantity_electrolyte = dict_cfg["quantity_electrolyte"]
        , ligand = Compound(**dict_cfg["ligand"])
        , metal = Compound(**dict_cfg["metal"])
        
    )

def load_cfg(
    dict_cfg: str
) -> RunCfg:
    return RunCfg (
        # potentiostat = load_cfg_pot(dict_cfg["potentiostat"])
        CV = load_CV_cfg(dict_cfg["CV"])
        , DPV = load_DPV_cfg(dict_cfg["DPV"])
        , experiment = load_cfg_exp(dict_cfg["experiment"])
    )

def load_ref_cfg(
    dict_cfg: str
) -> RefCfg:
    return RefCfg (
        # potentiostat = load_cfg_pot(dict_cfg["potentiostat"])
        CV = load_CV_cfg(dict_cfg["CV"])
        , DPV = load_DPV_cfg(dict_cfg["DPV"])
)


from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import sys
from scipy.signal import find_peaks

def process_cycles(
    csv_path: Path
):
    global reduction, oxidation
    HEADER = ("I", "V", "R", "T", "h", "t", "c")
    df = pd.read_csv(
        csv_path
        , names=HEADER
        , index_col=False
    )
    df = df[-0.75 < df["V"]]
    df = df[df["V"] < 0.75]

    df["c"] = df["c"].astype(int)
    # Get Red Ox
    (_, reduction), (_, oxidation) = df.groupby(np.diff(df["V"].array, append=0) > 0)
    oxi_peaks = map(
        lambda xs: xs[1][["V", "I"]].iloc[find_peaks(xs[1]["I"], width=10, distance=10, rel_height=0.9)[0]]
        , oxidation.groupby("c")
    )
    red_peaks = map(
        lambda xs: xs[1][["V", "I"]].iloc[find_peaks(-xs[1]["I"], width=10, distance=10, rel_height=0.2)[0]]
        , reduction.groupby("c")
    )
    return tuple(oxi_peaks), tuple(red_peaks)


async def get_csv_string(ar):
    stream = StringIO("")
    np.savetxt(stream, ar, delimiter=',')
    stream.seek(0)
    return stream.read()

def get_csv_string2(ar):
    stream = StringIO("")
    np.savetxt(stream, ar, delimiter=',')
    stream.seek(0)
    return stream.read()



def log(instance, message, payload, ref):
    if ref:
        instance.send_intermediate_response(RunReference_IntermediateResponses(message, payload.encode("utf-8")))
    else:
        instance.send_intermediate_response(RunExp_IntermediateResponses(message, payload.encode("utf-8")))


def run_CV(cfg):

    filename1 = str(uuid1())+"poten_1_cv.csv"
    cv_process_1 = subprocess.Popen(['python', '/home/poten/AsyncEchem/sila2_poten/feature_implementations/cv.py',
    str(cfg.CV.v_min),
    str(cfg.CV.v_max),
    str(cfg.CV.cycles), 
    str(cfg.CV.mV_s),
    str(cfg.CV.step_hz),
    str(cfg.CV.start_V),
    str(cfg.CV.last_V),
    str("/dev/poten_1"),
    str(filename1),
    ]
    )

    filename2 = str(uuid1())+"poten_2_cv.csv"
    cv_process_2 = subprocess.Popen(['python', '/home/poten/AsyncEchem/sila2_poten/feature_implementations/cv.py',
    str(cfg.CV.v_min),
    str(cfg.CV.v_max),
    str(cfg.CV.cycles), 
    str(cfg.CV.mV_s),
    str(cfg.CV.step_hz),
    str(cfg.CV.start_V),
    str(cfg.CV.last_V),
    str("/dev/poten_2"),
    str(filename2),
    ]
    )
    
    print("waiting")
    cv_process_2.wait()

    return filename1, filename2


def run_DPV(cfg):

    filename1 = "DPV"+str(uuid1())+"poten_1.csv"
    dpv_process_0 = subprocess.Popen(['python',
    '/home/poten/AsyncEchem/sila2_poten/feature_implementations/dpv.py',
    str(cfg.DPV.min_V),
    str(cfg.DPV.pulse_V),
    str(cfg.DPV.step_V),
    str(cfg.DPV.max_V),
    str(cfg.DPV.potential_hold_ms),
    str(cfg.DPV.pulse_hold_ms),
    str(cfg.DPV.voltage_hold_s),
    "/dev/poten_1",
    str(1),
    str(True),
    str(filename1),
    ])

    # filename2 = str(uuid1())+".csv"
    filename2 =  "DPV"+str(uuid1())+"poten_2.csv"
    dpv_process_1 = subprocess.Popen(['python',
    '/home/poten/AsyncEchem/sila2_poten/feature_implementations/dpv.py',
    str(cfg.DPV.min_V),
    str(cfg.DPV.pulse_V),
    str(cfg.DPV.step_V),
    str(cfg.DPV.max_V),
    str(cfg.DPV.potential_hold_ms),
    str(cfg.DPV.pulse_hold_ms),
    str(cfg.DPV.voltage_hold_s),
    "/dev/poten_2",
    str(1),
    str(True),
    str(filename2),
    ])
    
    dpv_process_1.wait()

    print("success dpv")

    return filename1, filename2


from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gaussian(x, A, mu, sigma, bkg):
    return bkg + A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gauss(data):
    max_current_pos = np.argmax(data[:, 2])

    init_guess = [data[max_current_pos, 2] - data[0, 2], data[max_current_pos, 1], np.std(data[:, 1]), data[0, 2]]
    try:
        gau_opt, gau_cov = curve_fit(gaussian, data[:, 1], data[:, 2], p0=init_guess)
        return gau_opt
    except:
        return init_guess


def proc_dpv(data, decay_ms:int = 500, pulse_ms:int = 20, pulse_from_end: int = 4, decay_from_end: int =50):
    drop_pts = decay_ms + pulse_ms
    num_periods = data.shape[0] // drop_pts

    dpv = np.empty((num_periods, 6))
    for i in range (0, num_periods):
        decay_end_point = drop_pts * i + decay_ms
        drop_end_point = drop_pts * (i + 1)
        dpv_time = data[decay_end_point-1, 0]
        dpv_cycle = data[decay_end_point-1, 3]
        dpv_exp = data[decay_end_point-1, 4]
        dpv_v_apply = data[decay_end_point-1, 5]
        dpv_current = np.mean(data[decay_end_point-decay_from_end:decay_end_point-1, 2]) - \
        np.mean(data[drop_end_point-pulse_from_end:drop_end_point-1, 2])
        dpv_voltage = np.mean(data[decay_end_point-decay_from_end:decay_end_point-1, 1])
        dpv[i] = [dpv_time, dpv_voltage, dpv_current, dpv_cycle, dpv_exp, dpv_v_apply]
    return dpv

def dpv_phasing(data):
    """
    split ONE cycle of dpv to positive and negative phase
    :param data:
    :return:
    """
    dpv_min = np.argmin(data[:, 5])
    dpv_max = np.argmax(data[:, 5])
    if data[2, 5] < data[0, 5]:
        phase_up = data[dpv_min:dpv_max, 0:5]
        phase_down = np.row_stack((data[0:dpv_min, 0:5], data[dpv_max:, 0:5]))
    else:
        phase_down = data[dpv_max:dpv_min, 0:5]
        phase_up = np.row_stack((data[0:dpv_max, 0:5], data[dpv_min:, 0:5]))
    return phase_up, phase_down

import sys


