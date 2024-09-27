from feature_implementations.utils  import (load_cfg, load_ref_cfg, load_CV_cfg, load_DPV_cfg,
                                            load_cfg_exp, proc_dpv, 
                                            dpv_phasing, fit_gauss, 
                                            gaussian)
from feature_implementations.exposed_feature import run_CV, run_CDPV, plot_cdpv, plot_cv, process_dpv_reference
from acp_wrapper import *
from prefect import task, flow, serve
from prefect.deployments import run_deployment
from prefect.futures import wait
import numpy as np
import json
import time 
from LabMind import FileObject, KnowledgeObject, nosql_service,cloud_service
from LabMind.Utils import upload
import os as os 
import asyncio
import concurrent.futures

@flow(log_prints=True)
def RunExp(Jobfile: str="jobfile.json"):
    """An experimental pipeline 
        complexation -> CDPV -> CV -> Rinse

    Args:
        Jobfile (str): Path to the jobfile.json. jobfile is a configuration file that contains the parameters for the experiment.

    Typically Jobfile looks like this:
        {
        "name": "test potentiostat job",
        "experiment": {
            "metal": {
            "position": 7,
            "volume": 0.5
            },
            "ligand": {
            "position": 9,
            "volume": 0.3
            },
            "quantity_buffer":0.4,
            "quantity_electrolyte":0.9,
            "num_mixings": 1
        },
        "DPV": {
            "min_V": 0,
            "pulse_V": 0.01,
            "step_V": 0.005,
            "max_V": 0.8,
            "voltage_hold_s": 0.1,
            "pulse_hold_ms": 100,
            "sample_hz": 250,
            "cycles": 1
        },
        "CV": {
            "v_min": -0.2,
            "v_max": 0.8,
            "cycles": 1,
            "mV_s": 200,
            "step_hz": 250,
            "start_V": null,
            "last_V": null
            }
        }

          
    Exanmple: 
    >>> RunExp("jobfile.json")
        RunExperiment Completed
    Prefect Deployment Example:
    >>> run_deployments(..) TODO: add example here 
    """
    autocomplex_client = create_autocomplex_client()
    with open("jobfile.json", "r") as jf:
        jobdict = json.load(jf)  
    cfg = load_cfg(jobdict)
    run_complexation(autocomplex_client, cfg)
    rxn_to_echem(autocomplex_client, 0)
    rxn_to_echem(autocomplex_client, 1)
    single_DPV(Jobfile,serial_port="/dev/poten_1")
    single_CV(Jobfile,serial_port="/dev/poten_1")
    Rinse()
    print("RunExperiment Completed")
    return "Completed"


@flow(log_prints=True)
def Rinse():

    """
    Executes the rinse procedure by performing the following steps:
    
    1. Creates an autocomplex client.
    2. Cleans the electrochemical cells (echem) for both cell 0 and cell 1.
    3. Cleans the reaction (rxn) chamber.
    
    This function ensures that the system is properly rinsed and ready for the next operation.
    
    Prints:
        "Rinse Completed" upon successful completion of the rinse procedure.
    
    Example:
    >>> Rinse()
        Rinse Completed
    """
    autocomplex_client = create_autocomplex_client()
    clean_echem(autocomplex_client, 0)
    clean_echem(autocomplex_client, 1)
    clean_rxn(autocomplex_client)
    print("Rinse Completed")
    return "Completed"

@flow(log_prints=True)
def RunReference(Jobfile):
    """
    Executes a reference run for the given job file.
    This function performs the following steps:
    1. Creates an autocomplex client.
    2. Loads the job configuration from a JSON file.
    3. Runs the complexation process.
    4. Converts the reference to electrochemical data.
    5. Runs CDPV (Cyclic Differential Pulse Voltammetry) on two serial ports and saves the results to CSV files.
    6. Processes the DPV reference data.
    7. Runs CV (Cyclic Voltammetry) on two serial ports and saves the results to CSV files.
    8. Rinses the system.
    9. Prints a completion message.
    Args:
        Jobfile (str): Path to the job file in JSON format.
    Returns:
        None
    Example:
    >>> RunReference("jobfile.json")
        RunReference Completed
    """

    autocomplex_client = create_autocomplex_client()
    with open("jobfile.json", "r") as jf:
        jobdict = json.load(jf) 
    name = jobdict["name"]
    cfg = load_ref_cfg(jobdict)
    run_complexation(autocomplex_client, cfg)
    ref_to_echem(autocomplex_client)
    DPV_ref_1:np.ndarray = single_DPV(Jobfile,serial_port="/dev/poten_1")
    process_dpv_reference(name, DPV_ref_1)

    CV_ref_1: np.ndarray = single_CV(Jobfile,serial_port="/dev/poten_1")
    Rinse()
    print("RunReference Completed")
    return "Completed"



@flow(log_prints=True)
def demo(Jobfile: str="jobfile.json"):
    """An experimental pipeline demo

    Args:
        Jobfile (str): Path to the jobfile.json. jobfile is a configuration file that contains the parameters for the experiment.

    Typically Jobfile looks like this:
        {
        "name": "test potentiostat job",
        "experiment": {
            "metal": {
            "position": 7,
            "volume": 0.5
            },
            "ligand": {
            "position": 9,
            "volume": 0.3
            },
            "quantity_buffer":0.4,
            "quantity_electrolyte":0.9,
            "num_mixings": 1
        },
        "DPV": {
            "min_V": 0,
            "pulse_V": 0.01,
            "step_V": 0.005,
            "max_V": 0.8,
            "voltage_hold_s": 0.1,
            "pulse_hold_ms": 100,
            "sample_hz": 250,
            "cycles": 1
        },
        "CV": {
            "v_min": -0.2,
            "v_max": 0.8,
            "cycles": 1,
            "mV_s": 200,
            "step_hz": 250,
            "start_V": null,
            "last_V": null
            }
        }

          
    Exanmple: 
    >>> demo("jobfile.json")
        RunExperiment Completed
    Prefect Deployment Example:
    >>> run_deployments(..) TODO: add example here 
    """
    autocomplex_client = create_autocomplex_client()
    with open("jobfile.json", "r") as jf:
        jobdict = json.load(jf)  
    cfg = load_cfg(jobdict)

    def main_thread():
        run_complexation(autocomplex_client, cfg) # 1
        rxn_to_echem(autocomplex_client, 0) # 1
    main_thread()

    # Block 1: DPV and CV for one potentiostat
    @task
    def block_1():
        run_deployment(name="single_DPV/single_DPV", parameters={"Jobfile": Jobfile, "serial_port": "/dev/poten_1"})
        run_deployment(name="single_CV/single_CV", parameters={"Jobfile": Jobfile, "serial_port": "/dev/poten_1"})

    @task
    def block_2():
        single_clean_rxn()
        run_complexation(autocomplex_client, cfg)
        rxn_to_echem(autocomplex_client, 1)
        run_deployment(name="single_DPV/single_DPV", parameters={"Jobfile": Jobfile, "serial_port": "/dev/poten_2"})
        run_deployment(name="single_CV/single_CV", parameters={"Jobfile": Jobfile, "serial_port": "/dev/poten_2"})

    blocks = [block_1.submit(), block_2.submit()]
    wait(blocks)
    # # Execute block_1 and block_2 in parallel using ThreadPoolExecutor
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     future_1 = executor.submit(block_1)
    #     future_2 = executor.submit(block_2)

    #     # Wait for both blocks to finish
    #     concurrent.futures.wait([future_1, future_2])

    Rinse()
    print("RunExperiment Completed")
    return "Completed"
# ------------------------ BASIC FLOWS ------------------------ #


@flow(log_prints=True)
def single_CV(Jobfile:str = "jobfile.json",serial_port="/dev/poten_1"):
    """
    Executes a single cyclic voltammetry (CV) experiment based on the provided job configuration file.
    Args:
        Jobfile (str): Path to the job configuration file in JSON format. Default is "jobfile.json".
        serial_port (str): Serial port to be used for the experiment. Default is "/dev/poten_1", te second channel would be "/dev/poten_2"
    Raises:
        FileNotFoundError: If the job configuration file is not found.
        json.JSONDecodeError: If the job configuration file is not a valid JSON.
        KeyError: If required keys are missing in the job configuration.
    Returns:
        None
    This function performs the following steps:
        1. Reads the job configuration from the specified JSON file.
        2. Loads the CV configuration using the `load_CV_cfg` function.
        3. Runs the CV experiment using the `run_CV` function.
        4. Saves the CV data to a CSV file.
        5. Plots the CV data and saves the plot as a PNG file.
        6. Uploads the CSV and PNG files to a cloud service.
        7. Deletes the local copies of the CSV and PNG files after uploading.
    Note:
        The function assumes the existence of several external functions and classes:
        - `load_CV_cfg`: Function to load CV configuration.
        - `run_CV`: Function to run the CV experiment.
        - `plot_cv`: Function to plot the CV data.
        - `FileObject`: Class to handle file metadata and uploading.
        - `upload`: Function to upload files to a cloud service.
        - `cloud_service` and `nosql_service`: Instances of services used for uploading and metadata storage.
    """

    with open(Jobfile, "r") as jf:
        jobdict = json.load(jf)  
    name = jobdict["name"]
    cfg = load_cfg(jobdict)
    CV_0: np.ndarray = run_CV(cfg, serial_port=serial_port)
    file_name = f"{name}_CV_poten_1.csv"
    np.savetxt(f"{name}_CV_poten_1.csv", CV_0, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")
    
    
    print("RunExperiment CV Completed on port ",serial_port)
    
    plot_name = f"{name}_CV_poten_1.png"
    plot_cv(CV_0, plot_name)
    # upload and save the file 
    csv_metadata = {
        "filename": file_name, 
        "project": "SDL_Test",
        "collection": "Potentialstat_Result", 
        "experiment_type": "CV",
        "data_type": "csv",
        "parameters": jobdict,
        "folder_structure": ["project","collection"],
        "description": "CV test result for SDL experiment",
    }
    file = FileObject(file_name, csv_metadata, cloud_service, nosql_service, embedding = False)
    upload(file)
    file.delete_local_file()
    
    plot_metadata = {
        "filename": plot_name,
        "project": "SDL_Test",
        "collection": "Potentialstat_Result",
        "experiment_type": "CV",
        "data_type": "png",
        "parameters": jobdict,
        "folder_structure": ["project","collection"],
        "description": "CV test plot for SDL experiment",
    }
    plot_file = FileObject(plot_name, plot_metadata, cloud_service, nosql_service, embedding = False)
    upload(plot_file)
    plot_file.delete_local_file()
    return CV_0



@flow(log_prints=True)
def single_DPV(Jobfile:str = "jobfile.json",serial_port="/dev/poten_1"):
    with open(Jobfile, "r") as jf:
        jobdict = json.load(jf)  
    name = jobdict["name"]
    cfg = load_cfg(jobdict)
    DPV_0: np.ndarray = run_CDPV(cfg, serial_port=serial_port)
    file_name = f"{name}_DPV_poten_1.csv"
    np.savetxt(file_name, DPV_0, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")
    print("RunExperiment DPV Completed on port ", serial_port)
    #plot_name = f"{name}_DPV_poten_1.png"
    #plot_cdpv(DPV_0, plot_name, do_fit=True)
    
    # upload and save the file 
    csv_metadata = {
        "filename": file_name, 
        "project": "SDL_Test",
        "collection": "Potentialstat_Result", 
        "experiment_type": "DPV",
        "data_type": "csv",
        "parameters": jobdict,
        "folder_structure": ["project","collection"],
        "description": "DPV test result for SDL experiment",
    }
    file = FileObject(file_name, csv_metadata, cloud_service, nosql_service, embedding = False)
    upload(file)
    file.delete_local_file()
    
    # plot_metadata = {
    #     "filename": plot_name,
    #     "project": "SDL_Test",
    #     "collection": "Potentialstat_Result",
    #     "experiment_type": "DPV",
    #     "data_type": "png",
    #     "parameters": jobdict,
    #     "folder_structure": ["project","collection"],
    #     "description": "DPV test plot for SDL experiment",
    # }
    # plot_file = FileObject(plot_name, plot_metadata, cloud_service, nosql_service, embedding = False)
    # upload(plot_file)
    # plot_file.delete_local_file()
    return DPV_0


@flow(log_prints=True)
def single_clean_echem(chamber_id:int = 0):
    """
    Perform a single electrochemical cleaning process for a specified chamber.
    This function initializes an autocomplex client and uses it to clean the 
    electrochemical chamber specified by the chamber_id.
    Args:
        chamber_id (int, optional): The ID of the chamber to be cleaned. Defaults to 0.
    Returns:
        None

    Example:
    >>> single_clean_echem(chamber_id=0)
    """

    autocomplex_client = create_autocomplex_client()
    clean_echem(autocomplex_client, chamber_id)


@flow(log_prints=True)
def single_clean_rxn():
    """
    Initializes an autocomplex client and performs a clean reaction.
    This function creates an autocomplex client using the `create_autocomplex_client` function
    and then calls the `clean_rxn` function with the created client to perform a cleaning reaction.
    Returns:
        None
    """
    autocomplex_client = create_autocomplex_client()
    clean_rxn(autocomplex_client)

@flow(log_prints=True)
def single_complexation(Jobfile:str):
    """
    Executes a single complexation process using the provided job file.
    Args:
        Jobfile (str): A JSON string representing the job configuration.
    This function performs the following steps:
    1. Creates an autocomplex client.
    2. Loads the job configuration from the provided JSON string.
    3. Runs the complexation process using the autocomplex client and the loaded configuration.
    Raises:
        json.JSONDecodeError: If the Jobfile is not a valid JSON string.
        KeyError: If required keys are missing in the job configuration.
    """
    autocomplex_client = create_autocomplex_client()
    with open(Jobfile, "r") as Jobfile:
        jobdict = json.load(Jobfile)
    cfg = load_cfg(jobdict)
    run_complexation(autocomplex_client, cfg)



if __name__ == "__main__":
    single_cv_deploy = single_CV.to_deployment(name="single_CV")
    single_dpv_deploy = single_DPV.to_deployment(name="single_DPV")
    rinse_deploy = Rinse.to_deployment(name="rinse_test")
    demo_deploy = demo.to_deployment(name="demo_test")  
    serve(single_cv_deploy,
          single_dpv_deploy,
          rinse_deploy,
          demo_deploy)