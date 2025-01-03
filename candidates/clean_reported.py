import pandas as pd
import numpy as np
import sys
from astropy.time import Time
import matplotlib.pyplot as plt

import psycopg2
import json
import json
import time
import requests

sys.path.append("../lib")
from alerce_tns import *

# access the database
credentials_file = "../../usecases/alercereaduser_v4.json"
with open(credentials_file) as jsonfile:
    params = json.load(jsonfile)["params"]
print("Opening connection to database...")
conn = psycopg2.connect(dbname=params['dbname'], user=params['user'], host=params['host'], password=params['password'])
print("Ready.")

#try:
#    # old database
#    credentials_file = "../../usecases/alercereaduser_v2.json"
#    with open(credentials_file) as jsonfile:
#        params = json.load(jsonfile)["params"]
#    conn_old = psycopg2.connect(dbname=params['dbname'], user=params['user'], host=params['host'], password=params['password'])
#except:
#    print("Cannot read old database")

# open the alerce client
client = alerce_tns()
print("Getting API key...")
api_key = open("../API.key", "r").read()

def get_DR(oid, url="https://api.alerce.online/ztf/dr/v1/light_curve/"):
    stats = client.query_object(oid, format='pandas')
    
    ra = float(stats.meanra)
    dec = float(stats.meandec)
    query = {'ra':ra, 'dec':dec, 'radius':1.5}

    response = requests.get(url, query)
    output = response.json()

    return output

def quantile(mags, mag):
    if mag < np.min(mags):
        return 0
    for idx, i in enumerate(np.sort(mags)):
        if mag > i:
            return ((idx + 1) / mags.shape[0])

ignore_index = []
DRg = {}
DRr = {}
mindetg = {}
mindetr = {}
percg = {}
percr = {}
bands = {1: "g", 2: "r"}
def alerce_reported(oid):

    time.sleep(1)

    # get candidate detections
    detections = client.query_detections(oid, format='pandas')
    for fid in detections.fid.unique():
        if "magpsf_corr" not in detections.loc[detections.fid == fid]:
            continue
        if detections.loc[detections.fid == fid].magpsf_corr.dropna().shape[0] == 0:
            continue
        
        minmag = detections.loc[detections.fid == fid].magpsf_corr.dropna().min()
        print(f"    Min mag {bands[fid]}: {minmag}")
        if fid == 1:
            mindetg[oid] = minmag
        if fid == 2:
            mindetr[oid] = minmag
            
    # get candidate forced photometry
    DR = get_DR(oid)
    for i in DR:
        if i["filterid"] == 1:
            if oid not in DRg.keys():
                DRg[oid] = np.array(i["mag"])
            else:
                DRg[oid] = np.concatenate([DRg[oid], np.array(i["mag"])])
        elif i["filterid"] == 2:
            if oid not in DRr.keys():
                DRr[oid] = np.array(i["mag"])
            else:
                DRr[oid] = np.concatenate([DRr[oid], np.array(i["mag"])])

    if oid in DRg.keys() and oid in mindetg.keys():
        print("    g DR quantile:", quantile(DRg[oid], mindetg[oid]))
    if oid in DRr.keys() and oid in mindetr.keys():
        print("    r DR quantile:", quantile(DRr[oid], mindetr[oid]))
    
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
df.set_index("object", inplace=True)
print(df.head())
print(df.shape)

# compute time difference wrt first report and last detection
print("Removing old reports")
now = Time.now()
df["now"] = df.apply(lambda row: Time.now().isot.replace(" ", "T")+"Z", axis=1).apply(pd.to_datetime)
df["first_detection_time"] =  df.first_detection.apply(lambda row: pd.to_datetime(row, utc=True))  # force UTC
df["delta_hours_first_detection"] = (df.first_detection_time - df.now).apply(lambda x: x.total_seconds()/3600)

df["last_date_time"] =  df.last_date.apply(lambda row: pd.to_datetime(row, utc=True)) # force UTC
df["delta_hours_last_date"] = (df.last_date_time - df.now).apply(lambda x: x.total_seconds()/3600)
# select only reports during the last 20 hours
df = df.loc[df.delta_hours_last_date > -20]
print(df.loc[df.delta_hours_last_date <= -20].index)
print(df.shape)

if df.shape[0] > 0:
    # sort by first detection (most recent first)
    df.sort_values(by=['first_detection_time'], ascending=False, inplace=True)
    
    # query ss_ztf
    print("Querying and removing known solar system objects")
    print(df.index)
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
else:
    print("No new reports in the last 20 hr")
    sys.exit()

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
    if oid in pp_new.index:
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
    print("    DR vs min alert mag quantiles:")
    for idx, i in enumerate(newcand):
        if i in DRg.keys() and i in mindetg.keys():
            qg = quantile(DRg[i], mindetg[i])
        else:
            qg = -1
        if i in DRr.keys() and i in mindetr.keys():
            qr = quantile(DRr[i], mindetr[i])
        else:
            qr = -1
        if qg == 0 or qr == 0:
            continue
        else:
            if qg > 0:
                print("   ", i, "g DR quantile:", "%.5f" % qg, ", min g mag stream: %.5f, min g mag DR: %.5f" % (mindetg[i], np.min(DRg[i])))
            if qr > 0:
                print("   ", i, "r DR quantile:", "%.5f" % qr, ", min r mag stream: %.5f, min r mag DR: %.5f" % (mindetr[i], np.min(DRr[i])))
            
    for idx in range(int(len(newcand)/nchunk) + 1):
        print("https://alerce.online/?" + "&".join(["oid=%s" % oid for oid in newcand[idx*nchunk:idx*nchunk+nchunk]]) + "&count=true&page=1&perPage=%i&sortDesc=false&selectedClassifier=stamp_classifier" % nchunk)
        
if len(oldcand) > 0:
    print("\nCandidates with first detections not in the last 20 hours:")
    for idx, i in enumerate(oldcand):
        print(idx, probstr(i), "https://alerce.online/object/%s" % i)
    print("    DR vs min alert mag quantiles:")
    for idx, i in enumerate(oldcand):
        if i in DRg.keys() and i in mindetg.keys():
            qg = quantile(DRg[i], mindetg[i])
        else:
            qg = -1
        if i in DRr.keys() and i in mindetr.keys():
            qr = quantile(DRr[i], mindetr[i])
        else:
            qr = -1
        if qg == 0 or qr == 0:
            continue
        else:
            if qg > 0:
                print("   ", i, "g DR quantile:", "%.5f" % qg, ", min g mag stream: %.5f, min g mag DR: %.5f" % (mindetg[i], np.min(DRg[i])))
            if qr > 0:
                print("   ", i, "r DR quantile", "%.5f" % qr, ", min r mag stream: %.5f, min g mag DR: %.5f" % (mindetr[i], np.min(DRr[i])))
    for idx in range(int(len(oldcand)/nchunk) + 1):
        print("https://alerce.online/?" + "&".join(["oid=%s" % oid for oid in oldcand[idx*nchunk:idx*nchunk+nchunk]]) + "&count=true&page=1&perPage=%i&sortDesc=falsee&selectedClassifier=stamp_classifier" % nchunk)


## plot comparison
#print("Plotting probabiliy comparison")
#probs_comp = pd.concat(probs_comp)
#probs_comp.set_index("oid", inplace=True)
#print(probs_comp)
#fig, ax = plt.subplots(figsize=(10, 10))
#if (probs_comp.index.isin(newcand)).sum() > 0:
#    ax.scatter(probs_comp.loc[probs_comp.index.isin(newcand)].prob_old, probs_comp.loc[probs_comp.index.isin(newcand)].prob_new, marker='o', s=40, c='r', label="first detected within the last 20 h")
#if (probs_comp.index.isin(oldcand)).sum() > 0:
#    ax.scatter(probs_comp.loc[probs_comp.index.isin(oldcand)].prob_old, probs_comp.loc[probs_comp.index.isin(oldcand)].prob_new, marker = 'o', c='k', label="first detected >20 h ago")
#probs_comp.apply(lambda row: ax.text(row.prob_old, row.prob_new, row.name), axis=1)
#ax.plot([0, 1], [0, 1], c='gray')
#ax.axhline(0.4, c='r')
#ax.axvline(0.4, c='r')
#ax.set_xlabel("prob_old")
#ax.set_ylabel("prob_new")
#plt.legend()
#plt.savefig("Comparison_%s.png" % sys.argv[1])
