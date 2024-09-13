from feature_implementations.utils  import (load_cfg, load_ref_cfg, load_CV_cfg, load_DPV_cfg,
                                            load_cfg_exp, proc_dpv, 
                                            dpv_phasing, fit_gauss, 
                                            gaussian)
from feature_implementations.exposed_feature import run_CV, run_CDPV
from acp_wrapper import *
from prefect import task, flow, serve
import numpy as np
import json
import time 
from LabMind import FileObject, KnowledgeObject, nosql_service,cloud_service
from LabMind.Utils import upload

@flow(log_prints=True)
def RunExp(Jobfile):
    autocomplex_client = create_autocomplex_client()
    with open("jobfile.json", "r") as Jobfile:
        jobdict = json.load(Jobfile)  
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


@flow(log_prints=True)
def Rinse():
    autocomplex_client = create_autocomplex_client()
    clean_echem(autocomplex_client, 0)
    clean_echem(autocomplex_client, 1)
    clean_rxn(autocomplex_client)
    print("Rinse Completed")

@flow(log_prints=True)
def RunReference(Jobfile):
    autocomplex_client = create_autocomplex_client()
    with open("jobfile.json", "r") as Jobfile:
        jobdict = json.load(Jobfile) 
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

@flow(log_prints=True)
def process_dpv_reference(Jobfile: str, DPV_data: np.ndarray):
    """Used after DPV measurments to process the data and save it as a reference
        from dpv.csv to fitcurv.csv
    Args:
        Jobfile (str): _description_
        filename (str): _description_
    """
    with open("jobfile.json", "r") as Jobfile:
        jobdict = json.load(Jobfile)  # Use json.load() instead of json.loads() for reading from a file
    name = jobdict["name"]
    dpv = proc_dpv(DPV_data, decay_ms =500, pulse_ms = 50, pulse_from_end=4,decay_from_end=20)
    np.savetxt(f'references/{name}_poten2.csv', dpv[:, 0:5], delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d")
    dpv_up, dpv_down = dpv_phasing(dpv)
    gau_opt = fit_gauss(dpv_up)
    with open(f"gau_opt.log", "a") as f:
        f.write(f"{name}_poten_2:\t"+str(gau_opt)+"\n")
    fit_curv = gaussian(dpv_up[:, 1], gau_opt[0], gau_opt[1], gau_opt[2], gau_opt[3])
    np.savetxt(f"references/{name}_poten2_fitcurv.csv", fit_curv, delimiter=',')


@flow(log_prints=True)
def single_CV(Jobfile:str = "jobfile.json",serial_port="/dev/poten_1"):
    with open(Jobfile, "r") as Jobfile:
        jobdict = json.load(Jobfile)  
    name = jobdict["name"]
    cfg = load_CV_cfg(jobdict["CV"])#TODO: need keys for others  cfg process, this should be fixed for loaders in the future, this also co changed the cfg.CV.xxx which CV are modified for testing purpose 
    CV_0: np.ndarray = run_CV(cfg, serial_port=serial_port)
    time.sleep(2)
    np.savetxt(f"{name}_CV_poten_1.csv", CV_0, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")
    print("RunExperiment CV Completed on port ",serial_port)
    csv_metadata = {
        "filename": f"{name}_CV_poten_1.csv", 
        "project": "SDL_Test",
        "collection": "Potentialstat_Result", 
        "experiment_type": "CV",
        "parameters": Jobfile,
        "folder_structure": ["project","collection"],
        "description": "CV test result for SDL experiment",
    }
    file = FileObject(f"{name}_CV_poten_1.csv", csv_metadata, cloud_service, nosql_service, embedding = False)
    upload(file)




@flow(log_prints=True)
def single_DPV(Jobfile:str = "jobfile.json",serial_port="/dev/poten_1"):
    with open(Jobfile, "r") as Jobfile:
        jobdict = json.load(Jobfile) 
    name = jobdict["name"]
    cfg = load_DPV_cfg(jobdict["DPV"])
    DPV_0: np.ndarray = run_CDPV(cfg, serial_port=serial_port)
    time.sleep(2)
    np.savetxt(f"{name}_DPV_poten_1.csv", DPV_0, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")
    print("RunExperiment DPV Completed on port ",serial_port)


@flow(log_prints=True)
def single_clean_echem(chamber_id:int = 0):
    autocomplex_client = create_autocomplex_client()
    clean_echem(autocomplex_client, chamber_id)


@flow(log_prints=True)
def single_clean_rxn():
    autocomplex_client = create_autocomplex_client()
    clean_rxn(autocomplex_client)

@flow(log_prints=True)
def single_complexation(Jobfile:str):
    autocomplex_client = create_autocomplex_client()
    jobdict = json.loads(Jobfile)
    cfg = load_cfg(jobdict)
    run_complexation(autocomplex_client, cfg)


if __name__ == "__main__":
    single_cv_deploy = single_CV.to_deployment(name="single_CV_test")
    single_dpv_deploy = single_DPV.to_deployment(name="single_DPV_test")
    single_clean_echem_deploy = single_clean_echem.to_deployment(name="single_clean_echem_test")
    single_clean_rxn_deploy = single_clean_rxn.to_deployment(name="single_clean_rxn_test")
    single_complexation_deploy = single_complexation.to_deployment(name="single_complexation_test")
    serve(single_cv_deploy,
          single_dpv_deploy,
          single_clean_echem_deploy,
          single_clean_rxn_deploy,
          single_complexation_deploy)