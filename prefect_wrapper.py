from feature_implementations.utils  import (load_cfg, load_ref_cfg, 
                                            load_cfg_exp, proc_dpv, 
                                            dpv_phasing, fit_gauss, 
                                            gaussian)
from feature_implementations.exposed_feature import run_CV, run_CDPV
from prefect import task, flow
from e_complex_robot import AutoComplex
import numpy as np
import json
import time


@flow(logs_print=True)
def RunExp(Jobfile):
    autocomplex_client = create_autocomplex_client()
    jobdict = json.loads(Jobfile)
    name = jobdict["name"]
    cfg = load_cfg(jobdict)
    run_complexation(autocomplex_client, cfg)
    rxn_to_echem(autocomplex_client, 0)
    rxn_to_echem(autocomplex_client, 1)
    DPV_0: np.ndarray = run_CDPV(cfg, serial_port="/dev/poten_1")
    DPV_1: np.ndarray = run_CDPV(cfg, serial_port="/dev/poten_2")
    time.sleep(2)
    np.savetxt(f"{name}_DPV_poten_1.csv", DPV_0, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")
    np.savetxt(f"{name}_DPV_poten_2.csv", DPV_1, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")

    CV_0: np.ndarray = run_CV(cfg, serial_port="/dev/poten_1")
    CV_1: np.ndarray = run_CV(cfg, serial_port="/dev/poten_2")
    time.sleep(2)
    np.savetxt(f"{name}_CV_poten_1.csv", CV_0, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")
    np.savetxt(f"{name}_CV_poten_2.csv", CV_1, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")
    Rinse()
    print("RunExperiment Completed")


@flow(logs_print=True)
def Rinse():
    autocomplex_client = create_autocomplex_client()
    clean_echem(autocomplex_client, 0)
    clean_echem(autocomplex_client, 1)
    clean_rxn(autocomplex_client)
    print("Rinse Completed")

@flow(logs_print=True)
def RunReference(Jobfile):
    autocomplex_client = create_autocomplex_client()
    jobdict = json.loads(Jobfile)
    name = jobdict["name"]
    cfg = load_ref_cfg(jobdict)
    run_complexation(autocomplex_client, cfg)
    ref_to_echem(autocomplex_client)
    DPV_ref_1: np.ndarray = run_CDPV(cfg, serial_port="/dev/poten_1")
    DPV_ref_2: np.ndarray = run_CDPV(cfg, serial_port="/dev/poten_2")
    time.sleep(2)
    np.savetxt(f"{name}_DPV_poten_1_ref.csv", DPV_ref_1, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")
    np.savetxt(f"{name}_DPV_poten_2_ref.csv", DPV_ref_2, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")
    process_dpv_reference(Jobfile, DPV_ref_1)
    process_dpv_reference(Jobfile, DPV_ref_2)

    CV_ref_1: np.ndarray = run_CV(cfg, serial_port="/dev/poten_1")
    CV_ref_2: np.ndarray = run_CV(cfg, serial_port="/dev/poten_2")
    time.sleep(2)
    np.savetxt(f"{name}_CV_poten_1_ref.csv", CV_ref_1, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")
    np.savetxt(f"{name}_CV_poten_2_ref.csv", CV_ref_2, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")

    Rinse()
    print("RunReference Completed")

@flow(logs_print=True)
def create_autocomplex_client():
    print("Creating AutoComplex client")
    client =  AutoComplex()
    print("AutoComplex client created")
    return client


@flow(logs_print=True)
def run_complexation(client: AutoComplex, cfg):
    print("Running complexation")
    client.run_complexation(
        num_metal=cfg.experiment.metal.position,
        num_ligand=cfg.experiment.ligand.position,
        quantity_metal=cfg.experiment.metal.volume,
        quantity_ligand=cfg.experiment.ligand.volume,
        quantity_buffer=cfg.experiment.quantity_buffer,
        quantity_electrolyte=cfg.experiment.quantity_electrolyte,
        mix_iteration=cfg.experiment.num_mixings
        )
    print("Complexation finished")

@flow(logs_print=True)
def rxn_to_echem(client, channel_ID:int):
    client.rxn_to_echem(channel_ID)
    print(f"Product transferred to electrochemical cell {channel_ID}")


@flow(logs_print=True)
def clean_echem(client, channel_ID:int):
    print(f"Cleaning electrochemical cell {channel_ID}")
    client.clean_echem(channel_ID)
    print(f"Electrochemical cell {channel_ID} cleaned")

@flow(logs_print=True)
def clean_rxn(client):
    print("Cleaning reaction vessel")
    client.clean_rxn()
    print("Reaction vessel cleaned")

@flow(logs_print=True)
def ref_to_echem(client):
    print("Transferring reference to electrochemical cell")
    client.ref_to_echem()
    print("Reference transferred to electrochemical cell")

@flow(logs_print=True)
def process_dpv_reference(Jobfile: str, DPV_data: np.ndarray):
    """Used after DPV measurments to process the data and save it as a reference
        from dpv.csv to fitcurv.csv
    Args:
        Jobfile (str): _description_
        filename (str): _description_
    """
    jobdict =json.loads(Jobfile)
    name = jobdict["name"]
    dpv = proc_dpv(DPV_data, decay_ms =500, pulse_ms = 50, pulse_from_end=4,decay_from_end=20)
    np.savetxt(f'references/{name}_poten2.csv', dpv[:, 0:5], delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d")
    dpv_up, dpv_down = dpv_phasing(dpv)
    gau_opt = fit_gauss(dpv_up)
    with open(f"gau_opt.log", "a") as f:
        f.write(f"{name}_poten_2:\t"+str(gau_opt)+"\n")
    fit_curv = gaussian(dpv_up[:, 1], gau_opt[0], gau_opt[1], gau_opt[2], gau_opt[3])
    np.savetxt(f"references/{name}_poten2_fitcurv.csv", fit_curv, delimiter=',')




