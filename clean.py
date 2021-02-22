import sys, re

sys.path.append("./lib")
from alerce_tns import *

client = alerce_tns()

refstring = sys.argv[1]
candidates_dir = "candidates"
candidates = open("%s/%s.csv" % (candidates_dir, refstring), encoding='utf-8-sig').read().splitlines()

print(candidates, len(candidates))

print("Getting API key...")
api_key = open("API.key", "r").read()

ignore = []
for oid in candidates:
    
    if len(oid) > 12:
        oid = oid[-12:]

    print(oid)
    tns = client.get_tns(api_key, oid)

    if tns:
        print("Astronomical transient is known:", tns, "\n")
        info = client.get_tns_reporting_info(api_key, oid)
        print("Reporting info:", info)
        if info["reporter"] == "ALeRCE":
            ignore.append(oid)
        #if not info["internal_names"] is None:
        #    if oid in info["internal_names"]: # reported using ZTF internal name, do not report
        #        print("Object was reported using the same ZTF internal name")
        #        ignore.append(oid)
        #    else:
        #        print("Warning: No internal names were reported.")

print()
print("SNe to ignore:")
print(ignore)
