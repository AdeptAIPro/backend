def dummy_parser(response: str):
    # Convert fake JSON string into dict
    import json
    return json.loads(response)
