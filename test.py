import json
with open("jobfile.json", "r") as file:
    data = json.load(file)  # Use json.load() instead of json.loads() for reading from a file
    print(data)