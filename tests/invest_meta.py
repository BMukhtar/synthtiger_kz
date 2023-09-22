import os

from examples.custom.template import SynthTiger

# Define the new directory you want to change to
new_directory = '../.'

# Get the absolute path
abs_directory = os.path.abspath(new_directory)

# Check if the last segment of the path is 'synthtiger'
if os.path.basename(abs_directory) == 'synthtiger':
    # Change the current working directory
    os.chdir(new_directory)
    # Print the current working directory to verify the change
    print(os.getcwd())
elif os.path.basename(os.path.abspath("./")) == 'synthtiger':
    print("All fine no need to change")
else:
    print("The last segment of the path is not 'synthtiger'")
    raise Exception("Directory mismatch!")


import pprint

import synthtiger
import json

config_path = "./examples/custom/config_kz_no_augment.yaml"
output_path = "./results/invest"
input_meta_path = "./tests/input_meta.json"

synthtiger.set_global_random_seed(seed=0)

config = synthtiger.read_config(config_path)
pprint.pprint(config)
template = SynthTiger(config)

# Open the JSON file for reading
with open(input_meta_path, 'r') as json_file:
    # Use json.load() to load the JSON data into a dictionary
    input_meta = json.load(json_file)
data = template.generate_from_meta(input_meta=input_meta)

template.init_save(output_path)
template.save(output_path, data, 0)
template.end_save(output_path)
print("End save")
