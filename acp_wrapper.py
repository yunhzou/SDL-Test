from e_complex_robot import AutoComplex
from prefect import task, flow

@task
def create_autocomplex_client():
    print("Creating AutoComplex client")
    client =  AutoComplex()
    print("AutoComplex client created")
    return client


@task
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

@task
def rxn_to_echem(client, channel_ID:int):
    client.rxn_to_echem(channel_ID)
    print(f"Product transferred to electrochemical cell {channel_ID}")


@task
def clean_echem(client, channel_ID:int):
    print(f"Cleaning electrochemical cell {channel_ID}")
    client.clean_echem(channel_ID)
    print(f"Electrochemical cell {channel_ID} cleaned")

@task
def clean_rxn(client):
    print("Cleaning reaction vessel")
    client.clean_rxn()
    print("Reaction vessel cleaned")

@task
def ref_to_echem(client):
    print("Transferring reference to electrochemical cell")
    client.ref_to_echem()
    print("Reference transferred to electrochemical cell")



