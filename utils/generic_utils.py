

def load_text_data(path: str) -> str:
    with open(path, 'r') as file:
        data = file.read()
    return data
