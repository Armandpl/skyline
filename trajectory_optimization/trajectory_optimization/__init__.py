import pathlib

# Get the path to the current file (my_module.py)
current_file = pathlib.Path(__file__).resolve()

# Get the path to the directory containing the current file
current_dir = current_file.parent

# data dir
data_dir = current_dir.parent / "data"
