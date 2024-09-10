from prefect_wrapper import *
from prefect import flow, serve

@flow(log_prints=True)
def single_CV(Jobfile:str = "jobfile.json",serial_port="/dev/poten_1"):
    with open(Jobfile, "r") as Jobfile:
        jobdict = json.load(Jobfile)  # Use json.load() instead of json.loads() for reading from a file
    name = jobdict["name"]
    cfg = load_cfg(jobdict)
    CV_0: np.ndarray = run_CV(cfg, serial_port=serial_port)
    time.sleep(2)
    np.savetxt(f"{name}_CV_poten_1.csv", CV_0, delimiter=',', fmt="%.2E,%.2E,%.2E,%d,%d,%d")
    print("RunExperiment CV Completed on port ",serial_port)

@flow(log_prints=True)
def single_DPV(Jobfile:str = "jobfile.json",serial_port="/dev/poten_1"):
    with open(Jobfile, "r") as Jobfile:
        jobdict = json.load(Jobfile)  # Use json.load() instead of json.loads() for reading from a file
    name = jobdict["name"]
    cfg = load_cfg(jobdict)
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