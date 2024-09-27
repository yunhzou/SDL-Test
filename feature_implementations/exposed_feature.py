from feature_implementations.potentiostat import Potentiostat
from feature_implementations.proc_echem import *
from prefect import task, flow
import numpy as np
from LabMind import FileObject, KnowledgeObject, nosql_service,cloud_service
from LabMind.Utils import upload


def init_poten(serial_port: str,
               baudrate: int,
               device_ID: int):
    POTEN = Potentiostat(serial_port=serial_port, baudrate=baudrate, device_ID=device_ID)
    POTEN.connect()
    print("Connected to potentiostat serial_port: {}, baudrate: {}, device_ID: {}".format(serial_port, baudrate, device_ID))
    return POTEN


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

@task
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
                              cfg.DPV.voltage_hold_s, 
                              cfg.DPV.pulse_hold_ms, 
                              cfg.DPV.voltage_hold_s,
                              cfg.DPV.cycles)
    terminate_poten(POTEN_port)
    return dpv_result


@task
def plot_cv(cv_data, file_name: str):
    cv_raw = cv_data[:, 1:3]
    cv_avg = avg_vi(cv_raw, 16, 1)
    plt.scatter(cv_avg[:, 0], cv_avg[:, 1], s=1)
    plt.savefig(f"{file_name}")
    plt.close()
    return cv_avg

@task
def plot_cdpv(
        dpv_data, 
        file_name: str, 
        do_fit: bool, 
        log_file: str = None,
        decay_ms=500,
        pulse_ms=50
        ):
    dpv_proc = proc_dpv(dpv_data, decay_ms=decay_ms,pulse_ms=pulse_ms)
    dpv_up, dpv_down = dpv_phasing(dpv_proc)
    plt.scatter(dpv_up[:, 1], dpv_up[:, 2], c='b', s=5)
    plt.scatter(dpv_down[:, 1], dpv_down[:, 2], c='g', s=5)
    if do_fit:
        gau_opt = fit_gauss(dpv_up)
        with open(log_file, 'a') as f:
            f.write(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            for para in gau_opt:
                f.write(f",{para:.3E}")
            f.write("\n")
        fit_curv = gaussian(dpv_up[:, 1], gau_opt[0], gau_opt[1], gau_opt[2], gau_opt[3])
        plt.plot(dpv_up[:, 1], fit_curv, 'r-')
    plt.savefig(f"{file_name}")
    plt.close()
    #if (gau_opt[1] < 0.2) or (gau_opt[1] > 0.3) or (abs(gau_opt[2] > 0.1) or (gau_opt[3] > 2e-7):
    #    input("Bad electrode, check! Enter to continue")
    return dpv_proc


@task(log_prints=True)
def process_dpv_reference(name: str, DPV_data: np.ndarray):
    """
    Processes DPV (Differential Pulse Voltammetry) reference data based on the provided job file and DPV data.
    Args:
        name (str): name of the experiments
        DPV_data (np.ndarray): Numpy array containing the DPV data to be processed.
    Raises:
        FileNotFoundError: If the job file cannot be found.
        json.JSONDecodeError: If the job file is not a valid JSON.
    Notes:
        - The function reads the job file to extract the job name.
        - It processes the DPV data using the `proc_dpv` function with specific parameters.
        - The processed DPV data is saved to a CSV file named based on the job name.
        - The DPV data is then phased and fitted to a Gaussian curve.
        - The Gaussian optimization results are logged to a file.
        - The fitted curve data is saved to another CSV file.
    
    Example:
    >>> process_dpv_reference(name, DPV_data)
    """
    dpv = proc_dpv(DPV_data, decay_ms =500, pulse_ms = 50, pulse_from_end=4,decay_from_end=20)
    np.savetxt(f'references/{name}_poten2.csv', dpv[:, 0:5], delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d")
    dpv_up, dpv_down = dpv_phasing(dpv)
    gau_opt = fit_gauss(dpv_up)
    with open(f"gau_opt.log", "a") as f:
        f.write(f"{name}_poten_2:\t"+str(gau_opt)+"\n")
    fit_curv = gaussian(dpv_up[:, 1], gau_opt[0], gau_opt[1], gau_opt[2], gau_opt[3])
    np.savetxt(f"references/{name}_poten2_fitcurv.csv", fit_curv, delimiter=',')
    reference_csv_metadata = {
        "filename": f"{name}_poten2_fitcurv.csv",
        "project": "SDL_Test",
        "collection": "Potentialstat_Reference",
        "experiment_type": "DPV",
        "data_type": "csv",
        "folder_structure": ["project","collection"],
    }
    file = FileObject(f"references/{name}_poten2_fitcurv.csv", reference_csv_metadata, cloud_service, nosql_service, embedding = False)
    upload(file)
    file.delete_local_file()
