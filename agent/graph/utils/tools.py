import json
import yaml


def yaml_parser(json_data):
    load = json.dumps(json_data.dict())
    data = json.loads(load)

    # Convert the dictionary to a YAML string
    return yaml.dump(data, default_flow_style=False, sort_keys=False)
