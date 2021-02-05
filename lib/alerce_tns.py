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

from xml.etree import ElementTree
from io import BytesIO
from PIL import Image
import base64

from alerce.api import AlerceAPI

# add redshift to Simbad query
customSimbad = Simbad()
customSimbad.add_votable_fields('z_value')
customSimbad.TIMEOUT = 5 # 5 seconds

class alerce_tns(AlerceAPI):
    'module to interact with alerce api to send TNS report'
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.hosts = {} # known hosts not to query again in SDSS

    def start_aladin(self, survey="P/PanSTARRS/DR1/color-z-zg-g", layout_width=70, fov=0.025):
        'Start a pyaladin window (for jupyter notebooks) together with an information window'

        self.aladin = ipyal.Aladin(survey=survey, layout=Layout(width='%s%%' % layout_width), fov=fov)
        self.info = widgets.HTML()
        self.box_layout = Layout(display='flex', flex_flow='row', align_items='stretch', width='100%')
        self.box = Box(children=[self.aladin, self.info], layout=self.box_layout)
        display(self.box)

        
    def view_object(self, oid, ned=True, simbad=True, SDSSDR15=True, catsHTM=True, vizier=False):
        'display an object in aladin, query different catalogs and show their information when hovering with the mouse'
        
        # start aladin widget
        self.start_aladin()

        sn = self.get_stats(oid, format='pandas')
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
        if SDSSDR15:
            try:
                self.info.value= "Querying SDSS-DR15..."
                table_SDSSDR15 = self.get_SDSSDR15(float(sn.meanra), float(sn.meandec), 20.)
                if table_SDSSDR15:
                    table_SDSSDR15["cat_name"] = Column(["SDSSDR15"], name="cat_name")
                    self.aladin.add_table(table_SDSSDR15)
            except:
                print("Cannot query SDSS-DR15")
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

        
    def select_hosts(self, candidates, ned=True, simbad=True, SDSSDR15=True, catsHTM=True, vizier=False):
        'check a list of object ids using an iterator'

        # copy survey selections
        self.do_ned = ned
        self.do_simbad = simbad
        self.do_SDSSDR15 = SDSSDR15
        self.do_catsHTM = catsHTM
        self.do_vizier = vizier
        self.candidate_host_names = {}
        self.candidate_host_redshifts = {}
        self.candidate_iterator = iter(candidates)

        # iterate over candidates
        try:
            oid = next(self.candidate_iterator)
            self.current_oid = oid
        except StopIteration:
            del self.iterator
    
        self.view_object(oid, ned=self.do_ned, simbad=self.do_simbad, SDSSDR15=self.do_SDSSDR15, catsHTM=self.do_catsHTM, vizier=self.do_vizier)

        
    def process_objectClicked(self, data):
        'move to following object when clicking over an object'

        report_photoz = False
        # save clicked information
        if data["data"]["cat_name"] == "NED":
            self.candidate_host_names[self.current_oid] = data["data"]["Object Name"]
            if "Redshift" in data["data"].keys():
                isphotoz = False
                if "Redshift Flag" in data["data"].keys():
                    if data["data"]["Redshift Flag"] in ["PHOT"]:
                        isphotoz = True
                if report_photoz or not isphotoz:
                    self.candidate_host_redshifts[self.current_oid] = data["data"]["Redshift"]
        elif data["data"]["cat_name"] == "Simbad":
            self.candidate_host_names[self.current_oid] = data["data"]["MAIN_ID"]
            if "Z_VALUE" in data["data"].keys():
                self.candidate_host_redshifts[self.current_oid] = data["data"]["Z_VALUE"]
        elif data["data"]["cat_name"] == "SDSSDR15":
            self.candidate_host_names[self.current_oid] = self.hosts[data["data"]["objid"]]["host_name"]
            if "specz" in self.hosts[data["data"]["objid"]].keys():
                self.candidate_host_redshifts[self.current_oid] = self.hosts[data["data"]["objid"]]["specz"]
            elif "photoz" in self.hosts[data["data"]["objid"]].keys() and report_photoz:
                self.candidate_host_redshifts[self.current_oid] = self.hosts[data["data"]["objid"]]["photoz"]

        # move to next candidate
        try:
            if hasattr(self, "candidate_iterator"):
                oid = next(self.candidate_iterator)
                self.current_oid = oid
                self.view_object(oid, ned=self.do_ned, simbad=self.do_simbad, SDSSDR15=self.do_SDSSDR15, catsHTM=self.do_catsHTM, vizier=self.do_vizier)
        except StopIteration:
            self.info.value =  "<div><font size='5'>All candidates revised</font></div>"
        
    def process_objectHovered(self, data):
        
        if data["data"]["cat_name"] == "SDSSDR15":
            objid = data["data"]["objid"]
            self.info.value =  "Querying SDSSDR15 object %s..." % str(objid)
            if objid not in self.hosts.keys():
                self.hosts[objid] = self.get_SDSSDR15_redshift(objid)
            for k in self.hosts[objid].keys():
                data["data"][k] = self.hosts[objid][k]
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
    
    def get_SDSSDR15(self, ra, dec, radius):
        'get galaxy crossmatch from SDSS DR15 using SDSS DR15 API'
        
        params = {
            "ra": "%f" % ra,
            "dec": "%f" % dec,
            "sr": "%f" % (radius / 60.),
            "format": "csv"
        }
        SDSS_DR15_url = "http://skyserver.sdss.org/dr15/SkyServerWS"
        r = requests.get(url = "%s/ConeSearch/ConeSearchService" % SDSS_DR15_url, params = params, timeout=(2, 5))
        df = pd.read_csv(BytesIO(r.content), comment="#")
        df["objid"] = df["objid"].astype(str)
        mask = df.type == "GALAXY"
        if mask.sum() == 0:
            return
        
        vot = Table.from_pandas(df[mask])
        return vot

    def get_SDSSDR15_redshift(self, objid):
        'get galaxy redshift from SDSS DR15 using their explorer webpage (this should be changed to using their API or querying their database directly'

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
        stats = self.get_stats(oid, format='pandas')

        # RA, DEC
        RA, DEC = float(stats.meanra), float(stats.meandec)
        
        # discovery_datetime
        discovery_datetime = Time(float(stats.firstmjd), format='mjd').fits.replace("T", " ")

        # display discovery stamps
        self.plot_stamp(oid)

        # host name and redshift
        host_name = "NULL"
        host_redshift = "NULL"
        if oid in self.candidate_host_names.keys():
            host_name = self.candidate_host_names[oid]
        if oid in self.candidate_host_redshifts.keys():
            host_redshift = self.candidate_host_redshifts[oid]
            
        # get detections and non-detections
        if verbose:
                print("Getting detections and non-detections for object %s" % oid)
        detections = self.get_detections(oid, format='pandas')
        non_detections = self.get_non_detections(oid, format='pandas') # note that new API returns none if not non detections

        # display latest non-detections
        # filters: 110 (g-ZTF), 111 (r-ZTF), 112 (i-ZTF)
        filters_dict = {1: "110", 2:"111", 3:"112"}

        # boolean to note whether there are detections
        has_non_detections = True
        if non_detections is None:
            has_non_detections = False

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

        # check TNS
        if verbose:
            print("Doing TNS xmatches for object %s" % oid)
        tns = self.get_tns(api_key, oid)
        if tns:
            print("Astronomical transient is known:", tns, "\n")
            info = self.get_tns_reporting_info(api_key, oid)
            print("Reporting info:", info)
            if not info["internal_names"] is None:
                if oid in info["internal_names"]: # reported using ZTF internal name, do not report
                    print("Object was reported using the same ZTF internal name, do not report.")
                    return False
                else:
                    print("Object was not reported using the same ZTF internal name, report.")
            else:
                print("Warning: No internal names were reported.")
            #if int(tns[0]["objname"][:4]) > Time(stats.firstmjd, format='mjd').datetime.year - 4.: # match is within last 3 years, do not report
            #    if not test:
            #        return False
            #else: # match older than 3 years, report
            #    print("Match is from more than 3 years before, sending to TNS anyway...")

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
                    "error": "%s" % float(detections.iloc[0].sigmara),
                    "units": "arcsec"
                    },
                "dec": {
                    "value": "%s" % DEC,
                    "error": "%s" % float(detections.iloc[0].sigmadec),
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
        if oid != self.oid:
            self.get_stats(oid, format='pandas')
        
        url_tns_api="https://www.wis-tns.org/api/get"
        search_url=url_tns_api+'/search'
        search_obj=[("ra", "%f" % self.meanra), ("dec", "%f" % self.meandec), ("radius","5"), ("units","arcsec"),
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

    # get TNS object type
    def get_tns_reporting_info(self, api_key, oid):
        objname = self.get_tns(api_key, oid)[0]["objname"]
    
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
        reporter = response.json()['data']['reply']['reporting_group']['group_name'],
        internal_names = response.json()['data']['reply']['internal_names']
        try:
            internal_names = internal_names.split(", ")
        except:
            True

        return {"discoverer": group_name,
                "reporter": response.json()['data']['reply']['reporting_group']['group_name'],
                "internal_names": internal_names}
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

        
