import pandas as pd
import numpy as np
import sys
from astropy.time import Time
import matplotlib.pyplot as plt
from io import BytesIO
import re

from astropy import units as u
from astropy.coordinates import SkyCoord

import psycopg2
import json
import json
import time
import requests

sys.path.append("lib")
from alerce_tns import *

# read the input date
date = re.findall(".*(\d\d\d\d\d\d\d\d)", sys.argv[1])[0]
print(f"Date: {date}")


# open the alerce database
credentials_file = "../usecases/alercereaduser_v4.json"
with open(credentials_file) as jsonfile:
    params = json.load(jsonfile)["params"]
print("Opening connection to database...")
conn = psycopg2.connect(dbname=params['dbname'], user=params['user'], host=params['host'], password=params['password'])
print("Ready.")


# initialize the alerce client
client = alerce_tns()

# open the TNS API key
print("Getting TNS API key...")
api_key = open("API.key", "r").read()


# function to read ZTF data release information from the alerce database
def get_DR(oid, url="https://api.alerce.online/ztf/dr/v1/light_curve/"):
    try:
        stats = client.query_object(oid, format='pandas')
    except:
        time.sleep(1)
        stats = client.query_object(oid, format='pandas')
        
    ra = float(stats.meanra)
    dec = float(stats.meandec)
    query = {'ra':ra, 'dec':dec, 'radius':1.5}

    response = requests.get(url, query)
    output = response.json()

    df = []
    for i in output:
        aux = pd.DataFrame({'hmjd': np.array(i["hmjd"]), 'mag': np.array(i["mag"]), "magerr": np.array(i["magerr"])})
        aux["ID"] = int(i["_id"])
        aux["filterid"] = int(i["filterid"])
        aux["oid"] = oid
        df.append(aux)

    if df != []:  
        return pd.concat(df) #output
    else:
        return None

# function to read ZTF data release information from IPAC (recommended)
def get_ZTF_DR_full(IDs, url="https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?"):
    
    for idx, ID in enumerate(IDs):
        if idx == 0:
            url += f"ID={ID}"
        else:
            url += f"&ID={ID}"
    url += "&BAD_CATFLAGS_MASK=32768"
    url += "&FORMAT=CSV"
    #print(url)
    
    response = requests.get(url)
    output = pd.read_csv(BytesIO(response.content))
    
    return output


# function to get the quantile
def quantile(xs, x):
    for idx, i in enumerate(np.sort(xs)):
        if x < i:
            return (idx / xs.shape[0])
    return 1.


# process a given candidate
def alerce_reported(oid):

    # 1 s sleep to avoid API problems
    time.sleep(1)

    # get object coordinates
    try:
        stats = client.query_object(oid, format='pandas')
    except:
        time.sleep(1)
        stats = client.query_object(oid, format='pandas')

    ra = float(stats.meanra)
    dec = float(stats.meandec)

    # get candidate detections
    try:
        detections = client.query_detections(oid, format='pandas')
    except:
        time.sleep(1)
        detections = client.query_detections(oid, format='pandas')
        
    for fid in detections.fid.unique():
        if "magpsf_corr" not in detections.loc[detections.fid == fid]:
            continue
        if detections.loc[detections.fid == fid].magpsf_corr.dropna().shape[0] == 0:
            continue
        
        minmag = detections.loc[detections.fid == fid].magpsf_corr.dropna().min()
        print(f"    Min mag {bands[fid]}: {minmag}")
        mindet[oid] = {}
        mindet[oid][fid] = minmag
            
    # get candidate ZTF data release information
    DR = get_DR(oid)
    #print(DR)

    # get candidate ZTF full data release information
    if not DR is None: 
        IDs = DR.ID.unique()
        DRfull = get_ZTF_DR_full(IDs)
        DRfull["oid"] = oid
        #print(list(DRfull))

        quants[oid] = {}
        quantsfull[oid] = {}
        quantsfullpos[oid] = {}

        # check magnitude quantiles
        for filterid in DR.filterid.unique():
            if oid in mindet.keys():
                if filterid in mindet[oid].keys() and filterid in DR.filterid.unique():
                    quants[oid][filterid] = quantile(DR[DR.filterid == filterid].mag, mindet[oid][filterid])
                    print(f"    %s ZTF DR mag quantile: %f" % (bands[filterid], quants[oid][filterid]))
                if filterid in mindet[oid].keys() and filtercodes[filterid] in DRfull.filtercode.unique():
                    quantsfull[oid][filterid] = quantile(DRfull[DRfull.filtercode == filtercodes[filterid]].mag, mindet[oid][filterid])
                    print(f"    %s ZTF DR full mag quantile: %f" % (bands[filterid], quantsfull[oid][filterid]))
                    
        # check distance quantiles
        for filterid in quantsfull[oid].keys():
            mask = DRfull.filtercode == filtercodes[filterid]
            ras = DRfull.loc[mask].ra
            decs = DRfull.loc[mask].dec
            seps = SkyCoord(ra=ras*u.degree, dec=decs*u.degree, frame='icrs').separation(SkyCoord(ra=ras.mean()*u.degree, dec=decs.mean()*u.degree, frame='icrs'))
            seps = np.sort(np.array([float(sep / u.arcsec) for sep in seps]))
            sepobj = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs').separation(SkyCoord(ra=ras.mean()*u.degree, dec=decs.mean()*u.degree, frame='icrs'))
            sepobj = float(sepobj / u.arcsec)
            quantsfullpos[oid][filterid] = quantile(seps, sepobj)
            print(f"    %s ZTF DR full pos. quantile: %f" % (bands[filterid], quantsfullpos[oid][filterid]))
    else:
        DRfull = None

    # get TNS information
    try:
        tns = client.get_tns(api_key, oid)
    except:
        time.sleep(1)
        tns = client.get_tns(api_key, oid)

    if tns:
        print("Astronomical transient is known:", tns)
        try:
            info = client.get_tns_reporting_info(api_key, oid)
        except:
            time.sleep(1)
            info = client.get_tns_reporting_info(api_key, oid)
            
        print("Reporting info:", info)
        if "ALeRCE" in info["reporter"]:
            print("Ignoring %s" % oid)
            ignore_index.append(oid)
        if not info["internal_names"] is None:
            if oid in info["internal_names"]: # reported using ZTF internal name, do not report                                                                                                              
                print("Object was reported using the same ZTF internal name")

    print("\n")


# ------------------------

ignore_index = []
mindet = {}
quants = {}
quantsfull = {}
quantsfullpos = {}
bands = {1: "g", 2: "r"}
filtercodes = {1: "zg", 2: "zr"}

    
# read oids -------------------
print("\n\mReading candidate oids")
df = pd.read_csv(sys.argv[1])
df.set_index("object", inplace=True)
print(df.head())
print(df.shape)

# remove old objects -------------
print("\n\nRemoving old reports")
# compute time difference wrt first report and last detection
now = Time.now()
df["now"] = df.apply(lambda row: Time.now().isot.replace(" ", "T")+"Z", axis=1).apply(pd.to_datetime)

# for testing purposes
check_previous_date = False
# checking previous date
if check_previous_date:
  offset = -24.*22 # check according to date requested
  from datetime import date, datetime, timedelta
  df['now'] = df['now'] + timedelta(hours=offset)

# fill columns in the dataframe
df["first_detection_time"] =  df["first_detection"] + 'Z'
df["first_detection_time"] =  df.first_detection_time.apply(pd.to_datetime)
df["delta_hours_first_detection"] = (df.first_detection_time - df.now).apply(lambda x: x.total_seconds()/3600)
df["last_date_time"] = df.last_date.apply(pd.to_datetime)
df["last_detection_time"] = df.last_detection.apply(lambda time: pd.to_datetime(f"{time}Z"))
df["delta_hours_last_date"] = (df.last_date_time - df.now).apply(lambda x: x.total_seconds()/3600)
df["delta_hours_last_detection"] = (df.last_detection_time - df.now).apply(lambda x: x.total_seconds()/3600)

# select only objects with *reports* during the last 20 hours
deltamax = 20 # 20 hours
df = df.loc[df.delta_hours_last_date > -deltamax]
print(df.loc[df.delta_hours_last_date <= -deltamax].index)
print(df.shape)

# select only objects with *detections* within the last 48 hours
df = df.loc[df.delta_hours_last_detection > -48]
print(df.shape)

# continue if any objects left
if df.shape[0] > 0:
    
    # sort by first detection (most recent first)
    df.sort_values(by=['first_detection_time'], ascending=False, inplace=True)
    
    # query known solar system objects in ss_ztf
    print("\n\nQuerying and removing known solar system objects")
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
        print("\n\nNo known asteroids among the candidates")
else:
    print(f"\n\nNo new reports in the last {deltamax} hr")
    sys.exit()

# query stamp classifier probabilities in new database
print("\n\nQuerying classifier probabilities")
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
pp = pd.read_sql_query(query, conn)
pp.set_index("oid", inplace=True)
pp = pp[~pp.index.duplicated(keep='first')]
print("probs", pp.shape)

print("Comparing probabilities")
probs_comp = []
def probstr(oid):
    probs = ""
    if oid in pp.index:
        probs += "Prob: %.3f" % pp.loc[oid].probability
    return probs

# check candidates to ignore and clean the data frame
print("\n\nQuerying TNS and ignoring SNe with ZTF names")
nunique = len(df.index.unique())
for idx, i in enumerate(df.index.unique()):
    print(nunique - idx, i, probstr(i))
    # this is the main function to be called
    alerce_reported(i)
print("objects ignored:", ignore_index)
df = df.loc[~df.index.isin(ignore_index)]
print(df.shape)
    
# Select candidates with first detections in the last 20 hours
masknew = df.delta_hours_first_detection > -deltamax
dfnew = df.loc[masknew]

# count
print("\n\nChecking missed candidates by the two classifiers")
mask_new = dfnew.source == "SN Hunter 1.1-dev"
mask_old = dfnew.source == "SN Hunter"
both = dfnew.loc[mask_new].index[dfnew.loc[mask_new].index.isin(dfnew.loc[mask_old].index)]
newnotold = dfnew.loc[mask_new].index[~dfnew.loc[mask_new].index.isin(dfnew.loc[mask_old].index)]
oldnotnew = dfnew.loc[mask_old].index[~dfnew.loc[mask_old].index.isin(dfnew.loc[mask_new].index)]
newcand = np.array(df.loc[masknew].index.unique())
oldcand = np.array(df.loc[~masknew].index.unique())
print(newcand)
print(oldcand)

# loop among all new candidates
print("\n\nCheck all candidates:\n\n")

nchunk=1000
status = {}
print(f"\nCandidates with first detections in the last {deltamax} hours:")
if len(newcand) > 0:
    for idx, i in enumerate(newcand):
        print(idx, probstr(i), "https://alerce.online/object/%s" % i)
    print("    DR vs min alert mag quantiles:")
    for idx, i in enumerate(newcand):
        flag = False
        if i in quantsfull.keys():
            status[i] = ""
            for filterid in quantsfull[i].keys():
                if not flag and quantsfull[i][filterid] > 0:
                    flag = True
                if flag:
                    if quantsfullpos[i][filterid] < 0.975:
                        print(f" {i} DR {bands[filterid]} mag quantile: %.5f / %.5f (masked / not masked), DR {bands[filterid]} pos. quantile: %.5f (masked)." % (quantsfull[i][filterid], quants[i][filterid], quantsfullpos[i][filterid]))
                        status[i] = "DR"
                    else:
                        print(f" -> Offset? {i} DR {bands[filterid]} mag quantile: %.5f / %.5f (masked / not masked), DR {bands[filterid]} pos. quantile: %.5f (masked)." % (quantsfull[i][filterid], quants[i][filterid], quantsfullpos[i][filterid]))
                        status[i] = "DR (offset?)"
                else:
                    status[i] = ""
        else:
            status[i] = ""
    for idx in range(int(len(newcand)/nchunk) + 1):
        print("https://alerce.online/?" + "&".join(["oid=%s" % oid for oid in newcand[idx*nchunk:idx*nchunk+nchunk]]) + "&count=true&page=1&perPage=%i&sortDesc=false&selectedClassifier=stamp_classifier" % nchunk)

if len(oldcand) > 0:
    print("\nCandidates with first detections not in the last 20 hours:")
    for idx, i in enumerate(oldcand):
        print(idx, probstr(i), "https://alerce.online/object/%s" % i)
    print("    DR vs min alert mag quantiles:")
    for idx, i in enumerate(oldcand):
        flag = False
        if i in quantsfull.keys():
            status[i] = ""
            for filterid in quantsfull[i].keys():
                status[i] = ""
                if not flag and quantsfull[i][filterid] > 0:
                    flag = True
                if flag:
                    if quantsfullpos[i][filterid] < 0.975:
                        print(f" {i} DR {bands[filterid]} mag quantile: %.5f / %.5f (masked / not masked), DR {bands[filterid]} pos. quantile: %.5f (masked)." % (quantsfull[i][filterid], quants[i][filterid], quantsfullpos[i][filterid]))
                        status[i] = "DR"
                    else:
                        print(f" -> Offset? {i} DR {bands[filterid]} mag quantile: %.5f / %.5f (masked / not masked), DR {bands[filterid]} pos. quantile: %.5f (masked)." % (quantsfull[i][filterid], quants[i][filterid], quantsfullpos[i][filterid]))
                        status[i] = "DR (offset?)"
                else:
                    status[i] = ""
        else:
            status[i] = ""
    for idx in range(int(len(oldcand)/nchunk) + 1):
        print("https://alerce.online/?" + "&".join(["oid=%s" % oid for oid in oldcand[idx*nchunk:idx*nchunk+nchunk]]) + "&count=true&page=1&perPage=%i&sortDesc=false&selectedClassifier=stamp_classifier" % nchunk)

print('\n')

if len(newcand) > 0:
    for i in newcand:
        print('%s' % i)

if len(oldcand) > 0:
    for i in oldcand:
        print('%s' % i)

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


# check already rejected candidates
print("\nReading previously rejected candidates...")
#xls = pd.ExcelFile("https://docs.google.com/spreadsheets/d/e/2PACX-1vRFQaTkgXJUuwH7bcEsw6Q7QyraGzPc30-SprB2VymkDRElDMDAoRRj30BjGKtZAQs0NTiE1tJPMIT3/pub?output=xlsx")
xls = pd.ExcelFile("https://docs.google.com/spreadsheets/d/e/2PACX-1vSNxNp5PeK--gvBmpe78MxAmZmriF1VQrA7ibjL7ijeiKxnJsFQNDJuAvCxmd0tLa5B6igv0EYSxCrW/pub?output=xlsx")

def checkrow(row):
    st = [name[:2] for name in list(row.dropna())]
    if "DR" in st:
        return(row.name)


dfs = {}
rejected = {}
passed = {}
for tab in xls.sheet_names:
    if tab == "Satellites":
        continue
    print(tab)
    dfs[tab] = xls.parse(tab)
    dfs[tab].columns = dfs[tab].iloc[0].values
    dfs[tab].drop(0, inplace=True)
    dfs[tab] = dfs[tab].iloc[:, :5]
    dfs[tab].set_index("oid", drop=True, inplace=True)
    aux = dfs[tab].apply(lambda row: checkrow(row), axis=1)
    rejected[tab] = aux.index[aux.values != None]
    passed[tab] = aux.index[aux.values == None]
    
oids_passed = []
oids_rejected = []
for l in list(passed.values()):
    oids_passed.append(np.array(l.values))
oids_passed = np.concatenate(oids_passed)
for l in list(rejected.values()):
    oids_rejected.append(np.array(l.values))
oids_rejected = np.concatenate(oids_rejected)

def checkoids(oids, date_new, rejected):
    for date_old in sorted(rejected.keys())[::-1]:
        if int(date_new) <= int(date_old):
            continue
        for oid_new in oids:
            for oid_old in rejected[date_old]:
                if oid_new == oid_old:
                    print(f"WARNING: {oid_new} rejected in {date_old} ")
                    status[oid_new] = f"{status[oid_new]} - rejected on {date_old}"

# check if any of the candidates have already been rejected
checkoids(newcand, date, rejected)
checkoids(oldcand, date, rejected)


print(f"\n\nPreparing {date}.csv file...")
if not os.path.exists("candidates/drive"):
    os.makedirs("candidates/drive")
output = open(f"candidates/drive/{date}.csv", "w")
output.write(',"If bad candidate, annotate with CR (cosmic ray), SH (shape), BD (bad difference), DR (consistent with DR), SAT (satellite), HL (hostless), ID (likely star, AGN, etc. based on images, catalog ID)"\n')
output.write("oid,AM,FB,FF,GP,pipeline\n")
for key, val in status.items():
    print(key, val)
    output.write(f"{key},,,,,{val}\n")
output.close()


print("\n\n\n------------------------------------------------\n-------   PLEASE UPLOAD THE CANDIDATES TO DRIVE -----------------\n------- https://docs.google.com/spreadsheets/d/1YfZ0M5NhZIjiARPgDeb1UhtiljduT35spvX2T6sTKwE/edit?usp=sharing;  ---------\n\n--------------------------------------------------\n\n\n")
