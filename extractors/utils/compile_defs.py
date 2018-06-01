import json
import os


def run(defs_file):
    if not os.path.exists(defs_file):
        raise Exception("Definitions directory does not exist.")
    defs = json.load(open(defs_file, "r"))["definitions"]
    defs = [d for d in defs if d["enabled"]]
    return defs

if __name__ == "__main__":
    d = run("./defs.json")
    print (d)