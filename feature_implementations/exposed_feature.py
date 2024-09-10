from feature_implementations.potentiostat import Potentiostat
from prefect import task, flow
import numpy as np

@flow(log_prints=True)
def init_poten(serial_port: str,
               baudrate: int,
               device_ID: int):
    POTEN = Potentiostat(serial_port=serial_port, baudrate=baudrate, device_ID=device_ID)
    POTEN.connect()
    print("Connected to potentiostat serial_port: {}, baudrate: {}, device_ID: {}".format(serial_port, baudrate, device_ID))
    return POTEN

@flow(log_prints=True)
def perform_CV(POTEN: Potentiostat,
               v_min: float,
               v_max: float,
               cycles: int,
               mV_s: float,
               step_hz: int,
               start_V: float,
               last_V: float,)->np.ndarray:
    try:
        start_V = float(start_V)
    except:
        start_V = None

    try:
        last_V = float(start_V)
    except:
        last_V = None

    rtn = POTEN.perform_CV(
        float(v_min)
        , float(v_max)
        , int(cycles)
        , float(mV_s)
        , int(step_hz)
        , start_V
        , last_V
    )
    print("CV performed with parameters v_min: {}, v_max: {}, cycles: {}, mV_s: {}, step_hz: {}, start_V: {}, last_V: {}".format(v_min, v_max, cycles, mV_s, step_hz, start_V, last_V))
    return rtn

@flow(log_prints=True)
def perform_CDPV(POTEN: Potentiostat,
                 min_V: float,
                 pulse_V: float,
                 step_V: float,
                 max_V: float,
                 potential_hold_ms: float,
                 pulse_hold_ms: float,
                 voltage_hold_s: float,
                 filename: str)->np.ndarray:
    rtn = POTEN.perform_CDPV(
        min_V=float(min_V)
        , pulse_V=float(pulse_V)
        , step_V=float(step_V)
        , max_V=float(max_V)
        , potential_hold_ms=float(potential_hold_ms)
        , pulse_hold_ms=float(pulse_hold_ms)
        , voltage_hold_s=float(voltage_hold_s)
    )
    print("CDPV performed with parameters min_V: {}, pulse_V: {}, step_V: {}, max_V: {}, potential_hold_ms: {}, pulse_hold_ms: {}, voltage_hold_s: {}".format(min_V, pulse_V, step_V, max_V, potential_hold_ms, pulse_hold_ms, voltage_hold_s))
    return rtn


@flow(log_prints=True)
def terminate_poten(POTEN: Potentiostat):
    POTEN.write_switch(False)
    POTEN.disconnect()
    print("Disconnected from potentiostat")

@flow(log_prints=True)
def run_CV(cfg,
           serial_port:str = "/dev/poten_1"):
    POTEN_port = init_poten(serial_port=serial_port,
                               baudrate=115200, 
                               device_ID=2)
    
    cv_result = perform_CV(POTEN_port, 
                           cfg.CV.v_min, 
                           cfg.CV.v_max, 
                           cfg.CV.cycles, 
                           cfg.CV.mV_s, 
                           cfg.CV.step_hz, 
                           cfg.CV.start_V,
                           cfg.CV.last_V)
    terminate_poten(POTEN_port)
    return cv_result

@flow(log_prints=True)
def run_CDPV(cfg,
             serial_port:str = "/dev/poten_1"):
    POTEN_port = init_poten(serial_port=serial_port,
                               baudrate=115200, 
                               device_ID=2)
    dpv_result = perform_CDPV(POTEN_port, 
                              cfg.DPV.min_V, 
                              cfg.DPV.pulse_V, 
                              cfg.DPV.step_V, 
                              cfg.DPV.max_V, 
                              cfg.DPV.potential_hold_ms, 
                              cfg.DPV.pulse_hold_ms, 
                              cfg.DPV.voltage_hold_s)
    terminate_poten(POTEN_port)
    return dpv_result



