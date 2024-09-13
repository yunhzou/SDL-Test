from flows import single_CV, single_DPV
from LabMind import FileObject, KnowledgeObject, nosql_service,cloud_service
from LabMind.Utils import upload
from prefect import task, flow
#generate one random number 
random_number = 243234234

knowledge = {
    "project": "Tutorial",
    "collection": "knowledges",
    "unique_fields": ["content","some_number"],
    "content": "put something as parallel field or as dictionary, some_number ",
    "some_number": random_number,
    "new_11": 23123}
knowledge = KnowledgeObject(knowledge, nosql_service,embedding=False)
upload(knowledge)


#single_CV()
#single_DPV()