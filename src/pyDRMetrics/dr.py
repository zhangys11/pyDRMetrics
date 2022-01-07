from .pyDRMetrics import *
import sys
import json
import uuid 
import os

# Three params passed from C#
if __name__ == "__main__":
    csv = sys.argv[1]
    k = int(sys.argv[2])
    dr = sys.argv[3]    

    drm = DRMetrics.test(csv, k, dr)

    fn = os.path.dirname(os.path.realpath(__file__)) + "/" + str(uuid.uuid4()) + ".html"
    with open(fn, 'w') as f:
        f.write(drm.get_html())

    #print(os.getcwd())
    print(fn)
    #print(json.dumps(d))
    