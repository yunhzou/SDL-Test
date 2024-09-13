from feature_implementations.potentiostat import Potentiostat
from proc_echem import plot_cdpv, plot_cv
from prefect import task, flow
import numpy as np


@task
def init_poten(serial_port: str,
               baudrate: int,
               device_ID: int):
    POTEN = Potentiostat(serial_port=serial_port, baudrate=baudrate, device_ID=device_ID)
    POTEN.connect()
    print("Connected to potentiostat serial_port: {}, baudrate: {}, device_ID: {}".format(serial_port, baudrate, device_ID))
    return POTEN


@task
def perform_CV(POTEN: Potentiostat,
               v_min: float,
               v_max: float,
               cycles: int,
               mV_s: float,
               step_hz: int,
               start_V: float,
               last_V: float,)->np.ndarray:
    print('loading parameters')
    try:
        start_V = float(start_V)
    except:
        start_V = None

    try:
        last_V = float(start_V)
    except:
        last_V = None

    print('parameters loaded')
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


@task
def perform_CDPV(POTEN: Potentiostat,
                 min_V: float,
                 pulse_V: float,
                 step_V: float,
                 max_V: float,
                 potential_hold_ms: float,
                 pulse_hold_ms: float,
                 voltage_hold_s: float,
                 cycles: int)->np.ndarray:
    rtn = POTEN.perform_CDPV(
        min_V=float(min_V)
        , pulse_V=float(pulse_V)
        , step_V=float(step_V)
        , max_V=float(max_V)
        , potential_hold_ms=float(potential_hold_ms)
        , pulse_hold_ms=float(pulse_hold_ms)
        , voltage_hold_s=float(voltage_hold_s),
        cycles=int(cycles),
    )
    print("CDPV performed with parameters min_V: {}, pulse_V: {}, step_V: {}, max_V: {}, potential_hold_ms: {}, pulse_hold_ms: {}, voltage_hold_s: {}".format(min_V, pulse_V, step_V, max_V, potential_hold_ms, pulse_hold_ms, voltage_hold_s))
    return rtn



@task
def terminate_poten(POTEN: Potentiostat):
    POTEN.write_switch(False)
    POTEN.disconnect()
    print("Disconnected from potentiostat")


@task
def run_CV(cfg,
           serial_port:str = "/dev/poten_1"):
    """
    Wrap the CV experiment in a flow
    """    
    POTEN_port = init_poten(serial_port=serial_port,
                               baudrate=115200, 
                               device_ID=2)
    print(POTEN_port)
    #TODO: modify to cfg.cv.v_min, right now look like this because the cfg is not loaded properly (historical issues)
    cv_result = perform_CV(POTEN_port, 
                           cfg.v_min, 
                           cfg.v_max, 
                           cfg.cycles, 
                           cfg.mV_s, 
                           cfg.step_hz, 
                           cfg.start_V,
                           cfg.last_V)
    terminate_poten(POTEN_port)
    return cv_result

@task
def run_CDPV(cfg,
             serial_port:str = "/dev/poten_1"):
    POTEN_port = init_poten(serial_port=serial_port,
                               baudrate=115200, 
                               device_ID=2)
    dpv_result = perform_CDPV(POTEN_port, 
                              cfg.min_V, 
                              cfg.pulse_V, 
                              cfg.step_V, 
                              cfg.max_V, 
                              cfg.voltage_hold_s, 
                              cfg.pulse_hold_ms, 
                              cfg.voltage_hold_s,
                              cfg.cycles)
    terminate_poten(POTEN_port)
    return dpv_result


@task
def plot_cv(cv_data, file_name: str):
    plot_cv(cv_data, file_name)
    print("CV plot saved as ", file_name)

@task
def plot_cdpv(dpv_data, 
              file_name: str,
              do_fit: bool,
              log_file: str = None,
              decay_ms=500,
              pulse_ms=50):
    plot_cdpv(dpv_data=dpv_data, 
              file_name=file_name, 
              do_fit=do_fit, 
              log_file=log_file,
              decay_ms=decay_ms,
              pulse_ms=pulse_ms)
    print("CDPV plot saved as ", file_name)

