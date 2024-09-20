from flows import single_CV, single_DPV, Rinse, single_complexation

#single_CV()
#single_DPV()
#Rinse()
import json

jobdict = json.loads(Jobfile)
single_complexation("jobfile.json")