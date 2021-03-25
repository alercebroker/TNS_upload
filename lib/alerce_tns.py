import sys, re
from collections import OrderedDict

import numpy as np

import pandas as pd
from pandas.io.json import json_normalize

from datetime import datetime
from IPython.display import HTML

import astropy.units as u
from astropy import coordinates
from astropy.time import Time
from astropy.table import Table, Column

from astroquery.ned import Ned
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

import ipyaladin as ipyal
from ipywidgets import Layout, Box, widgets

import requests
import json
import time

from xml.etree import ElementTree
from io import BytesIO
from PIL import Image
import base64


from alerce.core import Alerce

#my_config = {
#    "ZTF_API_URL": "https://dev.api.alerce.online"
#}
#alerce.load_config_from_object(my_config)

#from alerce.api import AlerceAPI

# whether to report photozs to TNS and skyportal
report_photoz_TNS = False
report_photoz_skyportal = False

# add redshift to Simbad query
customSimbad = Simbad()
customSimbad.add_votable_fields('z_value')
customSimbad.add_votable_fields('rv_value')
customSimbad.add_votable_fields('rvz_type')
customSimbad.add_votable_fields('rvz_error')
customSimbad.add_votable_fields('rvz_qual')
customSimbad.TIMEOUT = 5 # 5 seconds

class alerce_tns(Alerce):
    'module to interact with alerce api to send TNS report'
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        # fix API address
        my_config = {
            "ZTF_API_URL": "https://dev.api.alerce.online"
        }
        self.load_config_from_object(my_config)
        
        self.hosts_queried = {} # known hosts not to query again in SDSS
        self.nremaining = 0 # already seen candidates

    def start_aladin(self, survey="P/PanSTARRS/DR1/color-z-zg-g", layout_width=70, fov=0.025):
        'Start a pyaladin window (for jupyter notebooks) together with an information window'

        self.aladin = ipyal.Aladin(survey=survey, layout=Layout(width='%s%%' % layout_width), fov=fov)
        self.info = widgets.HTML()
        self.box_layout = Layout(display='flex', flex_flow='row', align_items='stretch', width='100%')
        self.box = Box(children=[self.aladin, self.info], layout=self.box_layout)
        display(self.box)

        
    def view_object(self, oid, ned=True, simbad=True, SDSSDR16=True, catsHTM=True, vizier=False):
        'display an object in aladin, query different catalogs and show their information when hovering with the mouse'
        
        # start aladin widget
        self.start_aladin()

        sn = self.query_objects(oid=oid, format='pandas')
        self.aladin.target = "%f %f" % (sn.meanra, sn.meandec)
        co = coordinates.SkyCoord(ra=sn.meanra, dec=sn.meandec, unit=(u.deg, u.deg), frame='fk5')

        if ned:
            try:
                self.info.value = "Querying NED..."
                table_ned = Ned.query_region(co, radius=0.025 * u.deg) #0.02
                if table_ned:
                    table_ned["cat_name"] = Column(["NED"], name="cat_name")
                    self.aladin.add_table(table_ned)
            except:
                print("Cannot connect with NED...")
        if simbad:
            try:
                self.info.value = "Querying Simbad..."
                table_simbad = customSimbad.query_region(co, radius=0.01 * u.deg)#, equinox='J2000.0')   
                if table_simbad:
                    table_simbad["cat_name"] = Column(["Simbad"], name="cat_name")
                    self.aladin.add_table(table_simbad)
            except:
                print("Cannot connect with Simbad...")
        if SDSSDR16:
            try:
                self.info.value= "Querying SDSS-DR16..."
                table_SDSSDR16 = self.get_SDSSDR16(float(sn.meanra), float(sn.meandec), 20.)
                if table_SDSSDR16:
                    table_SDSSDR16["cat_name"] = Column(["SDSSDR16"], name="cat_name")
                    self.aladin.add_table(table_SDSSDR16)
            except:
                print("Cannot query SDSS-DR16")
        if catsHTM:
            try:
                self.info.value = "Querying catsHTM..."
                tables_catsHTM = self.catsHTM_conesearch(oid, 'all', 0.5)#20)
                if tables_catsHTM:
                    for key in tables_catsHTM.keys():
                        self.aladin.add_table(tables_catsHTM[key])
                        tables_catsHTM[key]["cat_name"] = Column(["catsHTM_%s" % key], name="cat_name")
            except:
                print("Cannot connect with catsHTM")
        if vizier:
            self.info.value = "Querying Vizier..."
            table_vizier = Vizier.query_region(co, radius=1. * u.arcsec)#0.01 * u.deg)
            if table_vizier:        
                for k in table_vizier.keys():
                    table_vizier[k]["cat_name"] = Column(["Vizier_%s" % k], name="cat_name")
                    self.aladin.add_table(table_vizier[k])
        self.info.value = ""
        table_cand = Table.from_pandas(pd.DataFrame(
                data={"ZTF_oid": [oid], "RA": [float(sn.meanra)], "DEC": [float(sn.meandec)], "cat_name": ["ZTF"]}))
        self.aladin.add_table(table_cand)

        self.aladin.add_listener('objectClicked', self.process_objectClicked)
        self.aladin.add_listener('objectHovered', self.process_objectHovered)

    def select_hosts(self, candidates, ned=True, simbad=True, SDSSDR16=True, catsHTM=True, vizier=False):
        'check a list of object ids using an iterator'

        # copy survey selections
        self.do_ned = ned
        self.do_simbad = simbad
        self.do_SDSSDR16 = SDSSDR16
        self.do_catsHTM = catsHTM
        self.do_vizier = vizier
        self.candidate_hosts = pd.DataFrame()
        self.candidate_iterator = iter(candidates)
        self.nremaining = len(candidates)

        # iterate over candidates
        try:
            oid = next(self.candidate_iterator)
            self.current_oid = oid
        except StopIteration:
            del self.iterator
    
        self.view_object(oid, ned=self.do_ned, simbad=self.do_simbad, SDSSDR16=self.do_SDSSDR16, catsHTM=self.do_catsHTM, vizier=self.do_vizier)

        
    def process_objectClicked(self, data):
        'move to following object when clicking over an object'

        # save clicked information
        if data["data"]["cat_name"] != "ZTF":
            candidate_host_source = data["data"]["cat_name"]
        if data["data"]["cat_name"] == "NED":
            candidate_host_ra = data["data"]["RA"]
            candidate_host_dec = data["data"]["DEC"]
            candidate_host_name = data["data"]["Object Name"]
            if "Redshift" in data["data"].keys():
                if "Redshift Flag" in data["data"].keys():
                    if data["data"]["Redshift"] in ["nan", "-99", "-999.0", "-9999.0"]:
                        print("Ignoring redshift")
                    else:
                        candidate_host_redshift = data["data"]["Redshift"]
                        if data["data"]["Redshift Flag"] in ["", "SPEC"]:
                            candidate_host_redshift_spec = True
                        else:
                            candidate_host_redshift_spec = False
                        candidate_host_redshift_type = data["data"]["Redshift Flag"]
        elif data["data"]["cat_name"] == "Simbad":
            coords = coordinates.SkyCoord("%s %s" % (data["data"]["RA"], data["data"]["DEC"]), unit=(u.hourangle, u.deg), frame='icrs')
            candidate_host_ra = coords.ra / u.deg
            candidate_host_dec = coords.dec / u.deg
            candidate_host_name = data["data"]["MAIN_ID"]
            if "Z_VALUE" in data["data"].keys():
                if not data["data"]["Z_VALUE"] in ["nan", "-99", "-999.0", "-9999.0"]:
                    candidate_host_redshift = data["data"]["Z_VALUE"]
                    if data["data"]["RVZ_ERROR"] != "nan": # based on experience, we only trust Simbad redshifts if they have an associated error
                        candidate_host_redshift_spec = True
                        if data["data"]["RVZ_TYPE"] == "z":
                            candidate_host_redshift_error = data["data"]["RVZ_ERROR"]
                        elif data["data"]["RVZ_TYPE"] == "v":
                            cspeed = 299792. # km/s
                            candidate_host_redshift_error = data["data"]["RVZ_ERROR"] / cspeed
                        else:
                            candidate_host_redshift_spec = False
                    else:
                        candidate_host_redshift_spec = False
                    if "RVZ_QUAL" in data["data"].keys():
                        candidate_host_redshift_type = data["data"]["RVZ_QUAL"]  # this assumes that Simbad only returns spectroscopic redshifts
        elif data["data"]["cat_name"] == "SDSSDR16":
            objid = data["data"]["objid"]
            candidate_host_name = self.hosts_queried[objid]["host_name"]
            candidate_host_ra = data["data"]["ra"]
            candidate_host_dec = data["data"]["dec"]
            if "specz" in self.hosts_queried[objid].keys():
                if self.hosts_queried[objid]["specz"] in ["nan", "-99", "-999.0", "-9999.0"]:
                    print("Ignoring redshift...")
                else:
                    candidate_host_redshift = self.hosts_queried[objid]["specz"]
                    candidate_host_redshift_spec = True
                    candidate_host_redshift_error = self.hosts_queried[objid]["specz_err"]
                    candidate_host_redshift_type = "specz"
            elif "photoz" in self.hosts_queried[objid].keys():
                if self.hosts_queried[objid]["photoz"] in ["nan", "-99", "-999.0", "-9999.0"]:
                    print("Ignoring redshift...")
                else:
                    candidate_host_redshift = self.hosts_queried[objid]["photoz"]
                    candidate_host_redshift_error = self.hosts_queried[objid]["photoz_err"]
                    candidate_host_redshift_spec = False
                    candidate_host_redshift_type = "photoz"

        if not "candidate_host_name" in locals():
            candidate_host_name = "NULL"
        if not "candidate_host_ra" in locals():
            candidate_host_ra = "NULL"
        if not "candidate_host_dec" in locals():
            candidate_host_dec = "NULL"
        if not "candidate_host_redshift" in locals():
            candidate_host_redshift = "NULL"
        if not "candidate_host_redshift_spec" in locals():
            candidate_host_redshift_spec = "NULL"
        if not "candidate_host_redshift_error" in locals():
            candidate_host_redshift_error = "NULL"
        if not "candidate_host_redshift_type" in locals():
            candidate_host_redshift_type = "NULL"
        if not "candidate_host_source" in locals():
            candidate_host_source = "NULL"

        # get candidate positionstats
        stats = self.query_objects(oid=self.current_oid, format='pandas')
        cand_ra, cand_dec = float(stats.meanra), float(stats.meandec)
        # compute offset to galaxy
        if candidate_host_ra != "NULL" and candidate_host_dec != "NULL":
            candidate_host_offset = coordinates.SkyCoord(candidate_host_ra * u.deg, candidate_host_dec * u.deg, frame='icrs').separation(
                    coordinates.SkyCoord(cand_ra * u.deg, cand_dec * u.deg, frame='icrs')).arcsecond
        else:
            candidate_host_offset = "NULL"

        newdf = pd.DataFrame([[candidate_host_name,
                               candidate_host_ra,
                               candidate_host_dec,
                               candidate_host_offset,
                               candidate_host_source,
                               candidate_host_redshift_spec,
                               candidate_host_redshift,
                               candidate_host_redshift_error,
                               candidate_host_redshift_type]],
                             columns = ["host_name", "host_ra", "host_dec", "host_offset", "host_source",
                                        "host_redshift_spec", "host_redshift", "host_redshift_error", "host_redshift_type"],
                             index = [self.current_oid])
        newdf.index.name = "oid"
        self.candidate_hosts = pd.concat([newdf, self.candidate_hosts])
        display(self.candidate_hosts.head(1))

        # move to next candidate
        try:
            if hasattr(self, "candidate_iterator"):
                self.nremaining -= 1
                print("%i candidates remaining" % self.nremaining)
                oid = next(self.candidate_iterator)
                self.current_oid = oid
                self.view_object(oid, ned=self.do_ned, simbad=self.do_simbad, SDSSDR16=self.do_SDSSDR16, catsHTM=self.do_catsHTM, vizier=self.do_vizier)
        except StopIteration:
            self.info.value =  "<div><font size='5'>All candidates revised</font></div>"
            print("\n\nSummary of host galaxies:")
            display(self.candidate_hosts)


    def process_objectHovered(self, data):
        
        if data["data"]["cat_name"] == "SDSSDR16":
            objid = data["data"]["objid"]
            self.info.value =  "Querying SDSSDR16 object %s..." % str(objid)
            if objid not in self.hosts_queried.keys():
                self.hosts_queried[objid] = self.get_SDSSDR16_redshift(objid)
            for k in self.hosts_queried[objid].keys():
                data["data"][k] = self.hosts_queried[objid][k]
        output = "<h2>%s</h2>" % self.current_oid
        output += "<h2>%s</h2>" % data["data"]["cat_name"]
        sel_keys = data["data"].keys()#["type", 'ra', 'dec']
        output += "<table style='width:100%''>" #"<table border=0 cellspacing=0 cellpadding=0>"
        for key in sel_keys:
            if key in data["data"].keys():
                if data["data"][key] in ['nan', '']:
                    continue
                if re.compile('.*redshift|specz|z_value.*', re.IGNORECASE).match(key):
                    output += "<tr><th><font size='3' color='red'>%s: %s</font></th></tr>" % (key, data["data"][key])
                if re.compile('.*photoz.*', re.IGNORECASE).match(key):
                    output += "<tr><th><font size='3' color='orange'>%s: %s</font></th></tr>" % (key, data["data"][key])
        for key in sel_keys:
            if key in data["data"].keys():
                if data["data"][key] in ['nan', '']:
                    continue
                if not re.compile('.*redshift.*', re.IGNORECASE).match(key) and not re.compile('.*photoz.*', re.IGNORECASE).match(key):
                    output += "<tr><th><font size='1>%s: %s</font></th></tr>" % (key, data["data"][key])
        output += "</table>"
        self.info.value =  '%s' % output
    
    def get_SDSSDR16(self, ra, dec, radius):
        'get galaxy crossmatch from SDSS DR16 using SDSS DR16 API'
        
        params = {
            "ra": "%f" % ra,
            "dec": "%f" % dec,
            "sr": "%f" % (radius / 60.),
            "format": "csv"
        }
        SDSS_DR16_url = "http://skyserver.sdss.org/dr16/SkyServerWS"
        r = requests.get(url = "%s/ConeSearch/ConeSearchService" % SDSS_DR16_url, params = params, timeout=(2, 5))
        df = pd.read_csv(BytesIO(r.content), comment="#")
        df["objid"] = df["objid"].astype(str)
        mask = df.type == "GALAXY"
        if mask.sum() == 0:
            return
        
        vot = Table.from_pandas(df[mask])
        return vot

    def get_SDSSDR15_redshift(self, objid):
        'get galaxy redshift from SDSS DR16 using their explorer webpage (this should be changed to using their API or querying their database directly'

        params = {
            'id': "%s" % objid
        }
        r = requests.get(url = "http://skyserver.sdss.org/dr15/en/tools/explore/obj.aspx", params = params)
        df = pd.read_html(str(r.content))
        results = {}

        # extract name and photoz
        for i in df:
            data = i.loc[0][0]
            if type(data) is str:
                m = re.match("(?P<host>SDSS\sJ.{18})", data[5:])
                if m:
                    results["host_name"] = m["host"]
            if data == 'Mjd-Date':
                if i.iloc[0][1] == 'photoZ (KD-tree method)':
                    photozs = i.iloc[1][1].split()
                    results["photoz"] = photozs[0]
                    results["photoz_err"] = photozs[2]
            if data == "Spectrograph":
                results["specz"] = i.iloc[2][1]
                results["specz_err"] = i.iloc[3][1]
        return results

    def get_SDSSDR16_redshift_spec(self, objid, mode='vot'):
        'get galaxy spectroscopic redshift from SDSS DR16 using SDSS DR16 API'
        
        SDSS_DR16_url = 'http://skyserver.sdss.org/dr16/SkyServerWS'
        sdss_query = '?cmd=select top 5 p.objid, s.z, s.zerr, s.class, s.zwarning from photoobj as p join specobj as s on s.bestobjid = p.objid where p.objid=%s&format=csv' % objid
        url = '%s/SearchTools/SqlSearch%s' % (SDSS_DR16_url, sdss_query)
        print(url)
        r = requests.get(url = url, timeout=(2, 5))
        df = pd.read_csv(BytesIO(r.content), comment="#")
        
        df['objid'] = df['objid'].astype(str)
        mask = df['class'] == 'GALAXY'
        if mask.sum() == 0:
            return

        if mode == 'vot':
            vot = Table.from_pandas(df[mask])
            return vot
        elif mode == 'pandas':
            return df[mask]

    def get_SDSSDR16_redshift_phot(self, objid, mode='vot'):
        'get galaxy photometric redshift from SDSS DR16 using SDSS DR16 API'
        
        SDSS_DR16_url = 'http://skyserver.sdss.org/dr16/SkyServerWS'
        sdss_query = '?cmd=select top 5 objid, z, zerr, photoerrorclass from photoz where objid=%s&format=csv' % objid
        url = '%s/SearchTools/SqlSearch%s' % (SDSS_DR16_url, sdss_query)
        print(url)
        r = requests.get(url = url, timeout=(2, 5))
        df = pd.read_csv(BytesIO(r.content), comment="#")
        
        df['objid'] = df['objid'].astype(str)

        if mode == 'vot':
            vot = Table.from_pandas(df)
            return vot
        elif mode == 'pandas':
            return df
        
    def get_SDSSDR16_redshift(self, objid):
        'get galaxy redshift from SDSS DR16 using their explorer webpage (this should be changed to using their API or querying their database directly'

        display(self.get_SDSSDR16_redshift_phot(objid))
        display(self.get_SDSSDR16_redshift_spec(objid))

        params = {
            'id': "%s" % objid
        }
        r = requests.get(url = "http://skyserver.sdss.org/dr16/en/tools/explore/obj.aspx", params = params)
        df = pd.read_html(str(r.content))
        results = {}

        # extract name and photoz
        for i in df:
            data = i.loc[0][0]
            if type(data) is str:
                m = re.match("(?P<host>SDSS\sJ.{18})", data[5:])
                if m:
                    results["host_name"] = m["host"]
            if data == 'Mjd-Date':
                if i.iloc[0][1] == 'photoZ (KD-tree method)':
                    photozs = i.iloc[1][1].split()
                    results["photoz"] = photozs[0]
                    results["photoz_err"] = photozs[2]
            if data == "Spectrograph":
                results["specz"] = i.iloc[2][1]
                results["specz_err"] = i.iloc[3][1]

        # if specz available query from quicklook
        if "specz" in results.keys():
            r = requests.get(url = "http://skyserver.sdss.org/dr16/en/tools/quicklook/summary.aspx", params = params)
            results["specz"] = float(re.findall("Redshift\s\(z\):.*\n.*>(\d.\d+)</td>", r.text)[0])
        
        return results

    # check if object was reported
    def isin_TNS(self, api_key, oid):

        # check TNS
        tns = self.get_tns(api_key, oid)
        if tns:
            print("Astronomical transient is known:", tns, "\n")
            info = self.get_tns_reporting_info(api_key, oid)
            print("Reporting info:", info)
            if not info["internal_names"] is None:
                if oid in info["internal_names"]: # reported using ZTF internal name, do not report
                    print("Object was reported using the same ZTF internal name, do not report.")
                    #return False
                else:
                    print("Object was not reported using the same ZTF internal name, report.")
            else:
                print("Warning: No internal names were reported.")

            #if int(tns[0]["objname"][:4]) > Time(stats.firstmjd, format='mjd').datetime.year - 4.: # match is within last 3 years, do not report
            #    if not test:
            #        return False
            #else: # match older than 3 years, report
            #    print("Match is from more than 3 years before, sending to TNS anyway...")

        return True

    # prepare TNS report
    def do_TNS_report(self, api_key, oid, reporter, verbose=False, test=False):

        print("http://alerce.online/object/%s" % oid)
        
        # ALeRCE groupid
        reporting_group_id = "74" # ALeRCE
        
        # ALeRCE groupid
        discovery_data_source_id = "48" # ZTF

        # internal name
        internal_name = oid
        
        # instrument
        instrument = "196" # P48 - ZTF-Cam, 104: P60 - P60-Cam
        
        # units
        inst_units = "1" #1: "ABMag"
        
        # at type
        at_type = "1" #1: "PSN" # PNS: probable supernova
                                
        # get stats
        if verbose:
            print("Getting stats for object %s" % oid)
        stats = self.query_objects(oid=oid, format='pandas')

        # RA, DEC
        RA, DEC = float(stats.meanra), float(stats.meandec)
        
        # discovery_datetime
        discovery_datetime = Time(float(stats.firstmjd), format='mjd').fits.replace("T", " ")

        # display discovery stamps
        #self.plot_stamp(oid)

        # host name and redshift
        host_name = self.candidate_hosts.loc[oid].host_name
        if report_photoz_TNS or self.candidate_hosts.loc[oid].host_redshift_spec:
            host_redshift = self.candidate_hosts.loc[oid].host_redshift
        else:
            host_redshift = "NULL"
            
        # get detections and non-detections
        if verbose:
                print("Getting detections and non-detections for object %s" % oid)
        detections = self.query_detections(oid, format='pandas')
        non_detections = self.query_non_detections(oid, format='pandas') # note that new API returns none if not non detections

        # display latest non-detections
        # filters: 110 (g-ZTF), 111 (r-ZTF), 112 (i-ZTF)
        filters_dict = {1: "110", 2:"111", 3:"112"}

        # boolean to note whether there are detections, change index if they exist
        has_non_detections = True
        if non_detections is None:
            has_non_detections = False
        else:
            non_detections.set_index("mjd", inplace=True)
        
        # get last non-detection
        if has_non_detections and non_detections.shape[0] > 0:

            mjd_last_non_det = non_detections[non_detections.index < float(stats.firstmjd)].index.max()
            if np.isfinite(mjd_last_non_det):
                if (non_detections.index == mjd_last_non_det).sum() > 1: # due to collisions in database!
                    diffmaglim = float(non_detections.loc[mjd_last_non_det].diffmaglim.iloc[0])
                    fid = int(non_detections.loc[mjd_last_non_det].fid.iloc[0])
                else:
                    diffmaglim = float(non_detections.loc[mjd_last_non_det].diffmaglim)
                    fid = int(non_detections.loc[mjd_last_non_det].fid)
                non_detection = {
                    "obsdate": "%s" % Time(float(mjd_last_non_det), format='mjd').fits.replace("T", " "),
                    "limiting_flux": "%s" % diffmaglim,
                    "flux_units": "%s" % inst_units, # instrument units (1 or ABMag???)
                    "filter_value": "%s" % filters_dict[fid], 
                    "instrument_value": "%s" % instrument,
                    "exptime": "30", # in the future, read from database
                    "observer": "Robot",
                    "comments": "Data provided by ZTF"
                    }
            else:
                #return False
                print("WARNING: non detections limits not available")
                has_non_detections = False
        else:
            #return False
            print("WARNING: non detections limits not available")
            has_non_detections = False
                
        # estimate dm/dt
        if has_non_detections:
            dmdtstr = ""
            for fid in sorted(detections.fid.unique()):
                if fid == 1:
                    fid_str = "g"
                elif fid == 2:
                    fid_str = "r"
                else:
                    continue
                mask_det = (detections.fid == fid)
                idx_1st_det = detections.loc[mask_det].mjd.idxmin()
                mask_non_det = (non_detections.index < float(stats.firstmjd)) & (non_detections.fid == fid)
                if mask_non_det.sum() > 1:
                    dm = np.array(detections.loc[idx_1st_det].magpsf - non_detections.loc[mask_non_det].diffmaglim)
                    dm_error = np.array(detections.loc[idx_1st_det].magpsf + detections.loc[idx_1st_det].sigmapsf - non_detections.loc[mask_non_det].diffmaglim)
                    dt = np.array(detections.loc[idx_1st_det].mjd - non_detections.loc[mask_non_det].index)
                    dmdt = np.array(dm/dt)
                    dmdt_error = np.array(dm_error/dt)
                    idx_dmdt = np.argmin(dmdt_error) #dmdt = (dm/dt).min()
                    dmdt_min = dmdt[idx_dmdt]
                    dmdt_error_min = dmdt_error[idx_dmdt]

                    # constrain dmdt if dt is too small and dmdt too large
                    if dt[idx_dmdt] < 0.5 and dmdt_min < -1:
                        print("WARNING: dt = %.5f days" % dt[idx_dmdt])
                        dmdt_min = -1.
                        if dmdtstr == "":
                            dmdtstr = " (%s-rise > %.2f mag/day" % (fid_str, abs(dmdt_min))
                        else:
                            dmdtstr = "%s, %s-rise > %.2f mag/day)" % (dmdtstr, fid_str, abs(dmdt_min))
                    elif dmdt_error_min < -0.05:
                        if dmdtstr == "":
                            dmdtstr = " (%s-rise > %.2f+-%.2f mag/day" % (fid_str, abs(dmdt_min), abs(dmdt_min) - abs(dmdt_error_min))
                        else:
                            dmdtstr = "%s, %s-rise > %.2f+-%.2f mag/day)" % (dmdtstr, fid_str, abs(dmdt_min), abs(dmdt_min) - abs(dmdt_error_min))
                    else:
                        dmdtstr = ""
            if dmdtstr != "":
                if dmdtstr[-1] != ")":
                    dmdtstr = "%s)" % dmdtstr
        else:
            non_detection = {
                "archiveid": "0",
                "archival_remarks": "ZTF non-detection limits not available"
            }


        # prepare remarks
        if has_non_detections and dmdtstr != "":
            remarks = "Early SN candidate%s classified using ALeRCE's stamp classifier - see http://arxiv.org/abs/2008.03309 - and the public ZTF stream. Discovery image and light curve in http://alerce.online/object/%s " % (dmdtstr, oid)
            #remarks = "Early SN candidate classified by ALeRCE using the public ZTF stream. Discovery image and light curve in http://alerce.online/object/%s " % (oid)
        else:
            remarks = "SN candidate classified using ALeRCE's stamp classifier - see http://arxiv.org/abs/2008.03309 - and the public ZTF stream. Discovery image and light curve in http://alerce.online/object/%s " % oid
        print(remarks)

        # check if object is in TNS
        if not self.isin_TNS(api_key, oid):
            return False
            
        # if any detection is negative skip this candidate
        if np.sum(detections.isdiffpos == -1) > 0:
            print(detections.isdiffpos)
            print("WARNING: there are %i negative detections, skipping candidate" % np.sum(detections.isdiffpos == -1))
            return False
        
        # photometry
        photometry = {"photometry_group": {}}
        for idx, candid in enumerate(detections.index):
            photometry["photometry_group"]["%i" % idx] = {
                "obsdate": "%s" % Time(float(detections.loc[candid].mjd), format='mjd').fits.replace("T", " "),
                "flux": "%s" % detections.loc[candid].magpsf,
                "flux_error": "%s" % detections.loc[candid].sigmapsf,
                "limiting_flux": "%s" % detections.loc[candid].diffmaglim,
                "flux_units": "%s" % inst_units,
                "filter_value": "%s" % filters_dict[detections.loc[candid].fid],
                "instrument_value": "%s" % instrument,
                "exptime": "30", # in the future, read from database
                "observer": "Robot",
                "comments": "Data provided by ZTF"
            }

        # fill all compulsory fields
        report = {
                "ra": {
                    "value": "%s" % RA,
                    "error": "%s" % float(stats.sigmara),
                    "units": "arcsec"
                    },
                "dec": {
                    "value": "%s" % DEC,
                    "error": "%s" % float(stats.sigmadec),
                    "units": "arcsec"
                    },
                "reporting_group_id": reporting_group_id,
                "discovery_data_source_id": discovery_data_source_id,
                "reporter": reporter,
                "discovery_datetime": discovery_datetime,
                "at_type": at_type,
                "internal_name": internal_name,
                "remarks": remarks,
                "non_detection": non_detection,
                "photometry": photometry
                }
        # fill optional fields
        if host_name != "NULL":
            report["host_name"] = host_name
        if host_redshift != "NULL":
            report["host_redshift"] = "%s" % host_redshift
        
        if verbose:
            print()
            print(report)
            print()
            
        return report


    def get_tns(self, api_key, oid):
        'get information about the candidate from TNS'
        
        # get ra, dec
        stats = self.query_objects(oid=oid, format='pandas')
        
        url_tns_api="https://www.wis-tns.org/api/get"
        search_url=url_tns_api+'/search'
        search_obj=[("ra", "%f" % stats.meanra), ("dec", "%f" % stats.meandec), ("radius","5"), ("units","arcsec"),
            ("objname",""), ("internal_name","")]
        search_obj=OrderedDict(search_obj)
        search_data=[('api_key',(None, api_key)), ('data',(None,json.dumps(search_obj)))]
        try:
            response=requests.post(search_url, files=search_data, timeout=(5, 10))
            reply = response.json()["data"]["reply"]
            if reply != []:
                return reply
            else:
                return False
        except Exception as e:
            return False

    def get_tns_reporting_info(self, api_key, oid):

        all_internal_names = []
        objnames = []
        discoverers = []
        reporters = []
        for obj in self.get_tns(api_key, oid): 
            objname = obj["objname"]
            objnames.append(objname)
    
            data = {
                "objname": objname
            }
    
            # get object type
            json_data = [('api_key', (None, api_key)),
                         ('data', (None, json.dumps(data)))]

            url_tns_api="https://www.wis-tns.org/api/get" #"https://wis-tns.weizmann.ac.il/api/get" 
            json_url = url_tns_api + '/object'
            response = requests.post(json_url, files = json_data)
            group_name = response.json()['data']['reply']['discovery_data_source']['group_name']
            discoverers.append(group_name)
            reporter = response.json()['data']['reply']['reporting_group']['group_name']
            reporters.append(reporter)
            internal_names = response.json()['data']['reply']['internal_names']
            try:
                internal_names = internal_names.split(", ")
                all_internal_names += internal_names
            except:
                True
        print(all_internal_names)

        return {"discoverer": discoverers, "objname": objnames,
                "reporter": reporters,
                "internal_names": all_internal_names}
        #try:
        #    object_type = response.json()["data"]["reply"]["object_type"]["name"]
        #except:
        #    object_type = None
        #
        #return objname, object_type
        

    # function for changing data to json format
    def format_to_json(self, source):
        # change data to json format and return
        parsed = json.loads(source, object_pairs_hook = OrderedDict)
        result = json.dumps(parsed, indent = 4)
        return result
    
    # function for sending json reports (AT or Classification)                                                                                                                                         
    def send_json_report(self, api_key, url, json_file_path):
        try:
            # url for sending json reports
            json_url = url + '/bulk-report'
            
            # read json data from file
            json_read = self.format_to_json(open(json_file_path).read())
            
            # construct list of (key,value) pairs
            json_data = [('api_key', (None, api_key)),
                       ('data', (None, json_read))]
            
            # send json report using request module
            response = requests.post(json_url, files = json_data)

            # return response
            return response
        
        except Exception as e:

            return [None,'Error message : \n'+str(e)]

    # SKYPORTAL
    # ----------------------------------

    # token based api
    def api(self, method, endpoint, token, data=None):
        headers = {'Authorization': f'token {token}'}
        response = requests.request(method, endpoint, json=data, headers=headers)
        return response

    # get filter ids
    def get_skyportal_filter_id(self, url, token):
        r = self.api('GET', url+"filters", token)
        idcands = {}
        print(f'HTTP code: {r.status_code}, {r.reason}')
        print()
        if r.status_code in (200, 400):
            for i in r.json()["data"]:
                idcands[i["name"]] = i["id"]
        return idcands["ALERCE"]

    # get filter ids
    def get_skyportal_group_id(self, url, token):
        r = self.api('GET', url+"groups", token)
        idsource = {}
        print(f'HTTP code: {r.status_code}, {r.reason}')
        print()
        if r.status_code in (200, 400):    
            for i in r.json()["data"]["user_groups"]:
                idsource[i["nickname"]] = i["id"]
        return idsource["ALERCE"]

    # check if candidate is in SkyPortal
    def isin_skyportal(self, url, token, oid):
        status = self.api('GET', url+"candidates/"+oid, token).json()["status"]
        return status == "success"
    
    
    # Prepare SkyPortal report
    def do_skyportal_report(self, url, token, oid, reporter, verbose=False, test=False):

        # check if candidate is in skyportal
        if self.isin_skyportal(url, token, oid):
            print("%s is in skyportal" % oid)
            return False
        
        # get ALeRCE stats
        if verbose:
            print("Getting stats for object %s" % oid)
        stats = self.get_objects(oid=oid, format='pandas')

        # build report
        report = {}
        report["candidates"] = {}
        report["candidates"]["ra"] = float(stats.meanra)
        report["candidates"]["dec"] = float(stats.meandec)
        report["candidates"]["id"] = oid            # Name of the object
        if self.candidate_hosts.loc[oid].host_redshift != "NULL" and (report_photoz_skyportal or self.candidate_hosts.loc[oid].host_redshift_spec):
            report["candidates"]["redshift"] = float(self.candidate_hosts.loc[oid].host_redshift)
        report["candidates"]["origin"] = "ZTF"        # Origin of the object.
        report["candidates"]["filter_ids"] = [self.get_skyportal_filter_id(url, token)]
        report["annotations"] = {}
        report["annotations"]["alerce"] = {}
        report["annotations"]["alerce"]["obj_type"] = "sn_candidate"
        report["annotations"]["alerce"]["obj_identification"] = "stamp_classifier+visual_inspection"
        if self.candidate_hosts.loc[oid].host_name != "NULL":
            report["annotations"]["alerce"]["host"] = {}
            report["annotations"]["alerce"]["host"]["name"] = self.candidate_hosts.loc[oid].host_name
            report["annotations"]["alerce"]["host"]["identification"] = "visual_inspection"
            if self.candidate_hosts.loc[oid].host_ra != "NULL" and self.candidate_hosts.loc[oid].host_dec != "NULL":
                host_ra = float(self.candidate_hosts.loc[oid].host_ra)
                host_dec = float(self.candidate_hosts.loc[oid].host_dec)
                report["annotations"]["alerce"]["host"]["ra"] = host_ra
                report["annotations"]["alerce"]["host"]["dec"] = host_dec
                if self.candidate_hosts.loc[oid].host_offset != "NULL":
                    report["annotations"]["alerce"]["host"]["offset_arcsec"] = self.candidate_hosts.loc[oid].host_offset # arcseconds
            if self.candidate_hosts.loc[oid].host_source != "NULL":
                report["annotations"]["alerce"]["host"]["source"] = self.candidate_hosts.loc[oid].host_source
            if self.candidate_hosts.loc[oid].host_redshift != "NULL":
                report["annotations"]["alerce"]["host"]["redshift"] = float(self.candidate_hosts.loc[oid].host_redshift)
            if self.candidate_hosts.loc[oid].host_redshift_error != "NULL":
                report["annotations"]["alerce"]["host"]["redshift_error"] = float(self.candidate_hosts.loc[oid].host_redshift_error)
            if self.candidate_hosts.loc[oid].host_redshift_type != "NULL":
                report["annotations"]["alerce"]["host"]["redshift_type"] = self.candidate_hosts.loc[oid].host_redshift_type
        report["annotations"]["alerce"]["reporters"] = reporter
        report["annotations"]["alerce"]["url"] = "https://alerce.online/object/%s" % oid

        return report

    # send to skyportal
    def send_skyportal_report(self, url, token, report):

        data = report["candidates"]
        obj_id = data["id"]
        data["passed_at"] = time.strftime('20%y-%m-%dT%H:%M:%S', time.gmtime())
        print(data)
        
        # report candidates
        r = self.api('POST', url+"candidates", token, data=data)
        
        print(f'HTTP code: {r.status_code}, {r.reason}')
        if r.status_code in (200, 400):
            display(r.json())
            return True
        else:
            print("Unable to post candidate")
            return False
    
        # report annotations
        if "annotations" in report.keys():
            data = {}
            data["obj_id"] = obj_id
            data["origin"] = "alerce"
            data["data"] = report["annotations"]
            data["group_ids"] = [self.get_skyportal_group_id(url, token)]
            print(data)
            
            r = self.api('POST', url+"annotations", token, data=data)
            
            print(f'HTTP code: {r.status_code}, {r.reason}')
            if r.status_code in (200, 400):
                display(r.json())
                return True
            else:
                print("Unable to post candidate")
                return False
