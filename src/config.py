import json


def load_configuration(file_path):
    f = open(file_path)

    data = json.load(f)

    f.close()

    return data
