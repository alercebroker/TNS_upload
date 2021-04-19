import pandas as pd
import numpy as np
import sys
from astropy.time import Time
import matplotlib.pyplot as plt

import psycopg2
import json
import json

sys.path.append("../lib")
from alerce_tns import *

# access the database
credentials_file = "../../usecases/alercereaduser_v4.json"
with open(credentials_file) as jsonfile:
    params = json.load(jsonfile)["params"]
conn = psycopg2.connect(dbname=params['dbname'], user=params['user'], host=params['host'], password=params['password'])

try:
    # old database
    credentials_file = "../../usecases/alercereaduser_v2.json"
    with open(credentials_file) as jsonfile:
        params = json.load(jsonfile)["params"]
    conn_old = psycopg2.connect(dbname=params['dbname'], user=params['user'], host=params['host'], password=params['password'])
except:
    print("Cannot read old database")

# open the alerce client
client = alerce_tns()
print("Getting API key...")
api_key = open("../API.key", "r").read()

ignore_index = []
def alerce_reported(oid):
    tns = client.get_tns(api_key, oid)
    
    if tns:
        print("Astronomical transient is known:", tns, "\n")
        info = client.get_tns_reporting_info(api_key, oid)
        print("Reporting info:", info)
        if "ALeRCE" in info["reporter"]:
            print("Ignoring %s" % oid)
            ignore_index.append(oid)
        if not info["internal_names"] is None:
            if oid in info["internal_names"]: # reported using ZTF internal name, do not report                                                                                                              
                print("Object was reported using the same ZTF internal name")


# read df
print("Reading input data frame")
df = pd.read_csv(sys.argv[1])
print(df.head())
df.set_index("object", inplace=True)
print(df.shape)

# compute time difference wrt first report and last detection
print("Removing old reports")
now = Time.now()
df["now"] = df.apply(lambda row: Time.now().isot.replace(" ", "T")+"Z", axis=1).apply(pd.to_datetime)
df["first_detection_time"] =  df.first_detection.apply(pd.to_datetime)
df["delta_hours_first_detection"] = (df.first_detection_time - df.now).apply(lambda x: x.total_seconds()/3600)

df["last_date_time"] =  df.last_date.apply(pd.to_datetime)
df["delta_hours_last_date"] = (df.last_date_time - df.now).apply(lambda x: x.total_seconds()/3600)
# select only reports during the last 20 hours
df = df.loc[df.delta_hours_last_date > -20]
print(df.loc[df.delta_hours_last_date <= -20].index)
print(df.shape)

# sort by first detection (most recent first)
df.sort_values(by=['first_detection_time'], ascending=False, inplace=True)

# query ss_ztf
print("Querying and removing known solar system objects")
query='''
SELECT oid, ssdistnr
FROM ss_ztf
WHERE
oid in (%s)
''' % ",".join(["'%s'" % oid for oid in df.index])
ss = pd.read_sql_query(query, conn)
ss.set_index("oid", inplace=True)
mask = (ss.ssdistnr != -999)
if mask.sum() > 0:
    print("Ignoring %i known asteroids" % mask.sum())
    ignore_index = ss.ssdistnr.loc[ss.ssdistnr != -999].index.tolist()
    df = df.loc[~df.index.isin(ignore_index)]
    print(ignore_index)
    print(df.shape)
else:
    print("No known asteroids among the candidates")

try:
    # query stamp classifier probabilities in old database
    print("Querying old classifier probabilities")
    query='''
    SELECT 
    oid, sn_prob
    FROM
    stamp_classification
    WHERE
    oid in (%s)
    ''' % ",".join(["'%s'" % oid for oid in df.index])
    pp_old = pd.read_sql_query(query, conn_old)
    pp_old.set_index("oid", inplace=True)
    print("old probs.", pp_old.shape)
except:
    pp_old = pd.DataFrame()
    print("Cannot read old database")

# query stamp classifier probabilities in new database
print("Querying new classifier probabilities")
query='''
SELECT 
oid, probability, classifier_version
FROM
 probability
WHERE
classifier_name = 'stamp_classifier'
AND
class_name = 'SN'
AND
oid in (%s)
''' % ",".join(["'%s'" % oid for oid in df.index])
pp_new = pd.read_sql_query(query, conn)
pp_new.set_index("oid", inplace=True)
pp_new = pp_new[~pp_new.index.duplicated(keep='first')]
print("new probs", pp_new.shape)

print("Comparing probabilities")
probs_comp = []
def probstr(oid):
    probs = ""
    if oid in pp_old.index and oid in pp_new.index:
        probs += "P_old: %.3f" % pp_old.loc[oid].sn_prob
        probs += ", P_new: %.3f" % pp_new.loc[oid].probability
        probs_comp.append(pd.DataFrame([{"prob_old": pp_old.loc[oid].sn_prob, "prob_new": pp_new.loc[oid].probability, "oid": oid}]))
    elif oid in pp_old.index:
        probs += "P_old: %.3f" % pp_old.loc[oid].sn_prob
    elif oid in pp_new.index:
        probs += "P_new: %.3f" % pp_new.loc[oid].probability
    return probs

# check candidates to ignore and clean the data frame
print("Querying TNS and ignoring SNe with ZTF names")
nunique = len(df.index.unique())
for idx, i in enumerate(df.index.unique()):
    print(nunique - idx, i, probstr(i))
    alerce_reported(i)
print("objects ignored:", ignore_index)
df = df.loc[~df.index.isin(ignore_index)]
print(df.shape)
    
# Select candidates with first detections in the last 20 hours
masknew = df.delta_hours_first_detection > -20
dfnew = df.loc[masknew]

# count
print("Checking missed candidates by the two classifiers")
mask_new = dfnew.source == "SN Hunter 1.1-dev"
mask_old = dfnew.source == "SN Hunter"
both = dfnew.loc[mask_new].index[dfnew.loc[mask_new].index.isin(dfnew.loc[mask_old].index)]
newnotold = dfnew.loc[mask_new].index[~dfnew.loc[mask_new].index.isin(dfnew.loc[mask_old].index)]
oldnotnew = dfnew.loc[mask_old].index[~dfnew.loc[mask_old].index.isin(dfnew.loc[mask_new].index)]
newcand = np.array(df.loc[masknew].index.unique())
oldcand = np.array(df.loc[~masknew].index.unique())
print(newcand)
print(oldcand)
print("\nCandidates with first detections in the last 20 hours:")
nchunk=1000
if len(newcand) > 0:
    for idx, i in enumerate(newcand):
        print(idx, probstr(i), "https://alerce.online/object/%s" % i)
    for idx in range(int(len(newcand)/nchunk) + 1):
        print("https://alerce.online/?" + "&".join(["oid=%s" % oid for oid in newcand[idx*nchunk:idx*nchunk+nchunk]]) + "&count=true&page=1&perPage=%i&sortDesc=false" % nchunk)
        
if len(oldcand) > 0:
    print("\nCandidates with first detections not in the last 20 hours:")
    for idx, i in enumerate(oldcand):
        print(idx, probstr(i), "https://alerce.online/object/%s" % i)
    for idx in range(int(len(oldcand)/nchunk) + 1):
        print("https://alerce.online/?" + "&".join(["oid=%s" % oid for oid in oldcand[idx*nchunk:idx*nchunk+nchunk]]) + "&count=true&page=1&perPage=%i&sortDesc=false" % nchunk)

# plot comparison
print("Plotting probabiliy comparison")
probs_comp = pd.concat(probs_comp)
probs_comp.set_index("oid", inplace=True)
print(probs_comp)
fig, ax = plt.subplots(figsize=(10, 10))
if (probs_comp.index.isin(newcand)).sum() > 0:
    ax.scatter(probs_comp.loc[probs_comp.index.isin(newcand)].prob_old, probs_comp.loc[probs_comp.index.isin(newcand)].prob_new, marker='o', s=40, c='r', label="first detected within the last 20 h")
if (probs_comp.index.isin(oldcand)).sum() > 0:
    ax.scatter(probs_comp.loc[probs_comp.index.isin(oldcand)].prob_old, probs_comp.loc[probs_comp.index.isin(oldcand)].prob_new, marker = 'o', c='k', label="first detected >20 h ago")
probs_comp.apply(lambda row: ax.text(row.prob_old, row.prob_new, row.name), axis=1)
ax.plot([0, 1], [0, 1], c='gray')
ax.axhline(0.4, c='r')
ax.axvline(0.4, c='r')
ax.set_xlabel("prob_old")
ax.set_ylabel("prob_new")
plt.legend()
plt.savefig("Comparison_%s.png" % sys.argv[1])
