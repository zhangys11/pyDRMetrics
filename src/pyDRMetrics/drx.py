from .pyDRMetrics import *
import sys
import json
import uuid 
import os

# Three params passed from C#
if __name__ == "__main__":
    csvX = sys.argv[1]
    csvZ = sys.argv[2]
    csvXr = sys.argv[3]

    drm = DRMetrics.from_files(csvX, csvZ, csvXr)

    fn = os.path.dirname(os.path.realpath(__file__)) + "/" + str(uuid.uuid4()) + ".html"
    with open(fn, 'w') as f:
        f.write(drm.get_html())

    #print(os.getcwd())
    print(fn)
    #print(json.dumps(d))
    