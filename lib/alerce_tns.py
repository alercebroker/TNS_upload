import sys, re, os
from collections import OrderedDict

import numpy as np

import pandas as pd
from pandas.io.json import json_normalize

import warnings

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
customSimbad.TIMEOUT = 10 # 5 seconds

# timeout for Ned
customNed = Ned()
customNed.TIMEOUT = 10

class alerce_tns(Alerce):
    'module to interact with alerce api to send TNS report'
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        ## fix API address
        #my_config = {
        #    "ZTF_API_URL": "https://dev.api.alerce.online"
        #}
        #self.load_config_from_object(my_config)
        tns_credentials_file = "tns_credentials.json"
        with open(tns_credentials_file) as jsonfile:
            tns_params = json.load(jsonfile)
        tns_id = tns_params["tns_id"]
        tns_name = tns_params["tns_name"]
        self.tns_headers = {'User-Agent': 'tns_marker{"tns_id":%s,"type": "bot", "name":"%s"}' % (tns_id, tns_name)}
        
        self.hosts_queried = {} # known hosts not to query again in SDSS

        self.bestguess = None

    def start_aladin(self, survey="P/PanSTARRS/DR1/color-z-zg-g", layout_width=70, fov=0.025):
        'Start a pyaladin window (for jupyter notebooks) together with an information window'

        self.aladin = ipyal.Aladin(survey=survey, layout=Layout(width='%s%%' % layout_width), fov=fov)
        self.info = widgets.HTML()
        self.box_layout = Layout(display='flex', flex_flow='row', align_items='stretch', width='100%')
        self.box = Box(children=[self.aladin, self.info], layout=self.box_layout)
        display(self.box)


    def find_bestguess_host(self, table_cand, table_bestguess, xmatches):

        newdf = []
        for key in xmatches.keys():
            for i in xmatches[key]:
                hostdata = self.process_data(i)
                if hostdata["candidate_host_ra"] != "NULL" and hostdata["candidate_host_dec"] != "NULL":
                    hostdata["candidate_host_offset"] = coordinates.SkyCoord(hostdata["candidate_host_ra"] * u.deg, hostdata["candidate_host_dec"] * u.deg, frame='icrs').separation(
                        coordinates.SkyCoord(float(table_cand["RA"]) * u.deg, float(table_cand["DEC"]) * u.deg, frame='icrs')).arcsecond
                    hostdata["candidate_host_offset_bestguess"] = coordinates.SkyCoord(hostdata["candidate_host_ra"] * u.deg, hostdata["candidate_host_dec"] * u.deg, frame='icrs').separation(
                        coordinates.SkyCoord(float(table_bestguess['ra']) * u.deg, float(table_bestguess["dec"]) * u.deg, frame='icrs')).arcsecond
                    hostdata["candidate_bestguess_offset"] = coordinates.SkyCoord(float(table_cand["RA"]) * u.deg, float(table_cand["DEC"]) * u.deg, frame='icrs').separation(
                        coordinates.SkyCoord(float(table_bestguess['ra']) * u.deg, float(table_bestguess["dec"]) * u.deg, frame='icrs')).arcsecond
                else:
                    hostdata["candidate_host_offset"] = "NULL"

                newdf.append(pd.DataFrame([[hostdata["candidate_host_name"],
                               hostdata["candidate_host_ra"],
                               hostdata["candidate_host_dec"],
                               hostdata["candidate_host_offset"],
                               hostdata["candidate_host_offset_bestguess"],
                               hostdata["candidate_bestguess_offset"],
                               hostdata["candidate_host_source"],
                               hostdata["candidate_host_redshift_spec"],
                               hostdata["candidate_host_redshift"],
                               hostdata["candidate_host_redshift_error"],
                               hostdata["candidate_host_redshift_type"]]],
                            columns = ["host_name", "host_ra", "host_dec", "host_offset", "host_offset_bestguess", "bestguess_offset", "host_source",
                                        "host_redshift_spec", "host_redshift", "host_redshift_error", "host_redshift_type"],
                             index = [str(table_cand["ZTF_oid"][0])]))

        newdf = pd.concat(newdf).sort_values("host_offset_bestguess")
        newdf.replace(["NULL", "nan", -99, -999, -9999], np.nan, inplace=True)
        newdf.index.name = "oid"

        # otherwise, the closest
        min_bestguess_offset = newdf.iloc[0].host_offset_bestguess
        min_host_offset = newdf.iloc[0].host_offset
        bestguess_offset = newdf.iloc[0].bestguess_offset
        nmin_offset = 4
        # spectroscopic redshifts with error
        mask_specz_err = newdf.host_redshift_spec & (newdf.host_redshift_error > 0)
        if mask_specz_err.sum() > 0:
            min_bestguess_offset_specz_err = newdf.loc[mask_specz_err].iloc[0].host_offset_bestguess
            min_offset_specz_err = newdf.loc[mask_specz_err].iloc[0].host_offset
            # the distance from the predicted position and from the candidate cannot be nmin_offset times the minimum distance from the predicted position and the candidate, respectively
            if (min_bestguess_offset_specz_err < nmin_offset * max(1, min_bestguess_offset)) and (min_offset_specz_err < nmin_offset * max(1, bestguess_offset)):
                return newdf.loc[mask_specz_err].iloc[0]
        # spectroscopic redshifts
        mask_specz = newdf.host_redshift_spec
        if mask_specz.sum() > 0:
            min_bestguess_offset_specz = newdf.loc[mask_specz].iloc[0].host_offset_bestguess
            min_offset_specz = newdf.loc[mask_specz].iloc[0].host_offset
            if (min_bestguess_offset_specz < nmin_offset * max(1, min_bestguess_offset)) and (min_offset_specz < nmin_offset * max(1, bestguess_offset)):
                return newdf.loc[mask_specz].iloc[0]
        # photometric redshifts
        mask_photoz = newdf.host_redshift_type == "photoz"
        if mask_photoz.sum() > 0:
            min_bestguess_offset_photoz = newdf.loc[mask_photoz].iloc[0].host_offset_bestguess
            min_offset_photoz = newdf.loc[mask_photoz].iloc[0].host_offset
            if (min_bestguess_offset_photoz < nmin_offset * max(1, min_bestguess_offset)) and (min_offset_photoz < nmin_offset * max(1, bestguess_offset)):
                return newdf.loc[mask_photoz].iloc[0]
        # if no redshift with previus conditions, check if nearest source if close enough to candidate
        if min_host_offset < nmin_offset * max(1, bestguess_offset):
            return newdf.iloc[0]
        else:
            print("WARNING: closest crossmatch is too far from the predicted position. No catalog crossmatches found.")
            display(newdf.fillna("NULL"))
            aux = pd.DataFrame(newdf.iloc[0])
            aux.loc[:] = np.nan
            return aux
        
    def view_object(self, oid, ned=True, simbad=True, SDSSDR16=True, catsHTM=True, vizier=False):
        'display an object in aladin, query different catalogs and show their information when hovering with the mouse'

        time.sleep(3)
        # start aladin widget
        self.start_aladin()

        sn = self.query_objects(oid=oid, format='pandas')
        self.aladin.target = "%f %f" % (sn.meanra, sn.meandec)
        co = coordinates.SkyCoord(ra=sn.meanra, dec=sn.meandec, unit=(u.deg, u.deg), frame='fk5')
            
        # ZTF candidate
        table_cand = Table.from_pandas(pd.DataFrame(
                data={"ZTF_oid": [oid], "RA": [float(sn.meanra)], "DEC": [float(sn.meandec)], "cat_name": ["ZTF"]}))
        self.aladin.add_table(table_cand)

        # best guess
        bestguess = False
        if not self.bestguess is None and self.dobestguess:
            if oid in self.bestguess.index.values:
                bestguess = True
        if bestguess:
            table_delight = Table()
            radelight = self.bestguess.loc[oid].ra
            decdelight = self.bestguess.loc[oid].dec
            table_delight["ra"] = [radelight * u.degree]
            table_delight["dec"] = [decdelight * u.degree]
            table_delight["cat_name"] = str("DELIGHT")
            table_delight["Object Name"] = oid
            cobestguess = coordinates.SkyCoord(ra=radelight, dec=decdelight, unit=(u.deg, u.deg), frame='fk5')
            # collect all data
            xmatches = {}
            if ned:
                table_ned = customNed.query_region(cobestguess, radius=0.01 * u.deg)
                if table_ned:
                    xmatches["NED"] = table_ned
                    xmatches["NED"]["cat_name"] = Column(["NED"], name="cat_name")
            if simbad:
                table_simbad = customSimbad.query_region(cobestguess, radius=0.01 * u.deg)#, equinox='J2000.0')   
                if table_simbad:
                    xmatches["Simbad"] = table_simbad
                    xmatches["Simbad"]["cat_name"] = Column(["Simbad"], name="cat_name")
            if SDSSDR16:
                table_SDSSDR16 = self.get_SDSSDR16(float(radelight), float(decdelight), 30.)
                if table_SDSSDR16:
                    xmatches["SDSSDR16"] = table_SDSSDR16
                    xmatches["SDSSDR16"]["cat_name"] = Column(["SDSSDR16"], name="cat_name")
            best = self.find_bestguess_host(table_cand, table_delight, xmatches)

            best = pd.DataFrame(best).transpose()
            best.index.name = "oid"
            display(best)

            if best.dropna(subset=['host_ra']).shape[0] > 0:

                table_best = Table()
                for i in best:
                    if i in ["host_ra", "host_dec"]:
                        table_best[i[5:]] = float(best[i]) * u.degree
                    else:
                        table_best[i] = [str(best[i].values)]
                table_best["cat_name"] = "bestguess"
            
                self.aladin.add_table(table_best)

            self.candidate_hosts = pd.concat([best, self.candidate_hosts])

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
                self.candidate_hosts.replace("nan", "NULL", inplace=True)
                self.candidate_hosts.replace(-99, "NULL", inplace=True)
                self.candidate_hosts.replace(-999, "NULL", inplace=True)
                self.candidate_hosts.replace(-9999, "NULL", inplace=True)
                self.candidate_hosts.fillna("NULL", inplace=True)
                self.candidate_hosts.reset_index(inplace=True)
                #self.candidate_hosts.drop_duplicates(inplace=True)
                self.candidate_hosts.set_index("oid", inplace=True)
                hostfile = "hosts/%s_hosts.csv" % self.refstring
                print("Saving hosts to %s" % hostfile)
                self.candidate_hosts.to_csv(hostfile)
                display(self.candidate_hosts)
            
            #if ned and "NED" in xmatches.keys():
            #    if xmatches["NED"]:
            #        self.aladin.add_table(xmatches["NED"])
            #if simbad and "Simbad" in xmatches.keys():
            #    if xmatches["Simbad"]:
            #        self.aladin.add_table(xmatches["Simbad"])
            #if SDSSDR16 and "SDSSDR16" in xmatches.keys():
            #    if xmatches["SDSSDR16"]:
            #        self.aladin.add_table(xmatches["SDSSDR16"])

        else:

            if ned:
                try:
                    self.info.value = "Querying NED..."
                    table_ned = customNed.query_region(co, radius=0.025 * u.deg)#0.025 * u.deg)
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
                    table_SDSSDR16 = self.get_SDSSDR16(float(sn.meanra), float(sn.meandec), 60.)#20.)
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

        self.aladin.add_listener('objectHovered', self.process_objectHovered)
        self.aladin.add_listener('objectClicked', self.process_objectClicked)

        hostfile = "hosts/%s_hosts.csv" % self.refstring
        print("Saving hosts to %s" % hostfile)
        self.candidate_hosts.to_csv(hostfile)
        os.system("beep -f 555 -l 460")


    def select_hosts(self, candidates, refstring, ned=True, simbad=True, SDSSDR16=True, catsHTM=True, vizier=False, dobestguess=True):
        'check a list of object ids using an iterator'

        # copy survey selections
        self.do_ned = ned
        self.do_simbad = simbad
        self.do_SDSSDR16 = SDSSDR16
        self.do_catsHTM = catsHTM
        self.do_vizier = vizier
        self.candidate_iterator = iter(candidates)
        self.nremaining = len(candidates)
        self.dobestguess = dobestguess

        # save current galaxies
        if not os.path.exists("hosts"):
            os.makedirs("hosts")

        try:
            print("Loading and skipping already saved hosts...")
            print("hosts/%s_hosts.csv" % refstring)
            self.candidate_hosts = pd.read_csv("hosts/%s_hosts.csv" % refstring)
            self.candidate_hosts.reset_index(inplace=True)
            self.candidate_hosts.drop_duplicates(inplace=True)
            self.candidate_hosts.set_index("oid", inplace=True)
            print(self.candidate_hosts.shape)
            if not dobestguess:
                self.candidate_hosts.drop(candidates, inplace=True)
            self.candidate_hosts.fillna("NULL", inplace=True)
            display(self.candidate_hosts)
        except:
            print("Cannot load galaxy information, creating new information.")
            self.candidate_hosts = pd.DataFrame()

        # iterate over candidates
        try:
            oid = next(self.candidate_iterator)
            self.current_oid = oid
            self.refstring = refstring
            # in case we reload data skip oids
            while (oid in list(self.candidate_hosts.index)):
                try:
                    oid = next(self.candidate_iterator)
                except StopIteration:
                    del self.iterator
                print(oid)
                self.current_oid = oid
                self.nremaining -= 1
                if self.nremaining == 1:
                    print("All hosts recovered :)")
                    self.candidate_hosts.replace("nan", "NULL", inplace=True)
                    self.candidate_hosts.replace(-99, "NULL", inplace=True)
                    self.candidate_hosts.replace(-999, "NULL", inplace=True)
                    self.candidate_hosts.replace(-9999, "NULL", inplace=True)
                    self.candidate_hosts.fillna("NULL", inplace=True)
                    display(self.candidate_hosts)
                    return
        except StopIteration:
            del self.iterator

        self.view_object(oid, ned=self.do_ned, simbad=self.do_simbad, SDSSDR16=self.do_SDSSDR16, catsHTM=self.do_catsHTM, vizier=self.do_vizier)

    def process_data(self, data):

        hostdata = {}
        
        # save clicked information
        if data["cat_name"] != "ZTF":
            hostdata["candidate_host_source"] = data["cat_name"]
        if data["cat_name"] == "NED":
            hostdata["candidate_host_ra"] = data["RA"]
            hostdata["candidate_host_dec"] = data["DEC"]
            hostdata["candidate_host_name"] = data["Object Name"]
            hostdata["candidate_host_redshift_spec"] = False
            if "Redshift" in data.keys():
                if "Redshift Flag" in data.keys():
                    if not str(data["Redshift"]) in ["nan", "-99", "-999.0", "-9999.0", "--"]:
                        hostdata["candidate_host_redshift"] = data["Redshift"]
                        if data["Redshift Flag"] in ["", "SPEC"]:
                            hostdata["candidate_host_redshift_spec"] = True
                        hostdata["candidate_host_redshift_type"] = data["Redshift Flag"]
        elif data["cat_name"] == "Simbad":
            coords = coordinates.SkyCoord("%s %s" % (data["RA"], data["DEC"]), unit=(u.hourangle, u.deg), frame='icrs')
            hostdata["candidate_host_ra"] = coords.ra / u.deg
            hostdata["candidate_host_dec"] = coords.dec / u.deg
            hostdata["candidate_host_name"] = data["MAIN_ID"]
            hostdata["candidate_host_redshift_spec"] = False
            if "Z_VALUE" in data.keys():
                if not data["Z_VALUE"] in ["nan", "-99", "-999.0", "-9999.0"]:
                    hostdata["candidate_host_redshift"] = data["Z_VALUE"]
                    if data["RVZ_ERROR"] != "nan": # based on experience, we only trust Simbad redshifts if they have an associated error
                        hostdata["candidate_host_redshift_spec"] = True
                        if data["RVZ_TYPE"] == "z":
                            hostdata["candidate_host_redshift_error"] = data["RVZ_ERROR"]
                        elif data["RVZ_TYPE"] == "v":
                            cspeed = 299792. # km/s
                            hostdata["candidate_host_redshift_error"] = data["RVZ_ERROR"] / cspeed
                    if "RVZ_QUAL" in data.keys():
                        hostdata["candidate_host_redshift_type"] = data["RVZ_QUAL"]  # this assumes that Simbad only returns spectroscopic redshifts
        elif data["cat_name"] == "SDSSDR16":

            objid = data["objid"]
            
            # long name
            host_name =  "SDSS J%s" % coordinates.SkyCoord(ra=data["ra"]*u.degree, dec=data["dec"]*u.degree, frame='icrs').to_string(style="hmsdms", sep="", pad=True, precision=2)[:-1].replace(" ", "")
            self.info.value =  "Querying SDSSDR16 object %s..." % str(objid)
            if objid not in self.hosts_queried.keys():
                self.hosts_queried[objid] = self.get_SDSSDR16_redshift(objid)
                self.hosts_queried[objid]["host_name"] = host_name

            hostdata["candidate_host_name"] = self.hosts_queried[objid]["host_name"]
            hostdata["candidate_host_ra"] = data["ra"]
            hostdata["candidate_host_dec"] = data["dec"]

            if "specz" in self.hosts_queried[objid].keys():
                if self.hosts_queried[objid]["specz"] in ["nan", "-99", "-999.0", "-9999.0"]:
                    print("Ignoring redshift...")
                else:
                    hostdata["candidate_host_redshift"] = self.hosts_queried[objid]["specz"]
                    hostdata["candidate_host_redshift_spec"] = True
                    hostdata["candidate_host_redshift_error"] = self.hosts_queried[objid]["specz_err"]
                    hostdata["candidate_host_redshift_type"] = "specz"
            elif "photoz" in self.hosts_queried[objid].keys():
                if self.hosts_queried[objid]["photoz"] in ["nan", "-99", "-999.0", "-9999.0"]:
                    print("Ignoring redshift...")
                else:
                    hostdata["candidate_host_redshift"] = self.hosts_queried[objid]["photoz"]
                    hostdata["candidate_host_redshift_error"] = self.hosts_queried[objid]["photoz_err"]
                    hostdata["candidate_host_redshift_spec"] = False
                    hostdata["candidate_host_redshift_type"] = "photoz"
                    
        if not "candidate_host_name" in hostdata.keys():
            hostdata["candidate_host_name"] = "NULL"
        if not "candidate_host_ra" in hostdata.keys():
            hostdata["candidate_host_ra"] = "NULL"
        if not "candidate_host_dec" in hostdata.keys():
            hostdata["candidate_host_dec"] = "NULL"
        if not "candidate_host_redshift" in hostdata.keys():
            hostdata["candidate_host_redshift"] = "NULL"
        if not "candidate_host_redshift_spec" in hostdata.keys():
            hostdata["candidate_host_redshift_spec"] = "NULL"
        if not "candidate_host_redshift_error" in hostdata.keys():
            hostdata["candidate_host_redshift_error"] = "NULL"
        if not "candidate_host_redshift_type" in hostdata.keys():
            hostdata["candidate_host_redshift_type"] = "NULL"
        if not "candidate_host_source" in hostdata.keys():
            hostdata["candidate_host_source"] = "NULL"

        return hostdata
        
    def process_objectClicked(self, data):
        'move to following object when clicking over an object'

        hostdata = self.process_data(data["data"])

        # get candidate positionstats
        stats = self.query_objects(oid=self.current_oid, format='pandas')
        cand_ra, cand_dec = float(stats.meanra), float(stats.meandec)
        # compute offset to galaxy
        if hostdata["candidate_host_ra"] != "NULL" and hostdata["candidate_host_dec"] != "NULL":
            hostdata["candidate_host_offset"] = coordinates.SkyCoord(hostdata["candidate_host_ra"] * u.deg, hostdata["candidate_host_dec"] * u.deg, frame='icrs').separation(
                    coordinates.SkyCoord(cand_ra * u.deg, cand_dec * u.deg, frame='icrs')).arcsecond
        else:
            hostdata["candidate_host_offset"] = "NULL"

        newdf = pd.DataFrame([[hostdata["candidate_host_name"],
                               hostdata["candidate_host_ra"],
                               hostdata["candidate_host_dec"],
                               hostdata["candidate_host_offset"],
                               hostdata["candidate_host_source"],
                               hostdata["candidate_host_redshift_spec"],
                               hostdata["candidate_host_redshift"],
                               hostdata["candidate_host_redshift_error"],
                               hostdata["candidate_host_redshift_type"]]],
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
            self.candidate_hosts.replace("nan", "NULL", inplace=True)
            self.candidate_hosts.replace(-99, "NULL", inplace=True)
            self.candidate_hosts.replace(-999, "NULL", inplace=True)
            self.candidate_hosts.replace(-9999, "NULL", inplace=True)
            self.candidate_hosts.fillna("NULL", inplace=True)
            self.candidate_hosts.reset_index(inplace=True)
            self.candidate_hosts.drop_duplicates(inplace=True)
            self.candidate_hosts.set_index("oid", inplace=True)
            hostfile = "hosts/%s_hosts.csv" % self.refstring
            print("Saving hosts to %s" % hostfile)
            self.candidate_hosts.to_csv(hostfile)
            display(self.candidate_hosts)


    def process_objectHovered(self, data):

        if data["data"]["cat_name"] == "SDSSDR16":
            # objid
            objid = data["data"]["objid"]
            # long name
            host_name =  "SDSS J%s" % coordinates.SkyCoord(ra=data["data"]["ra"]*u.degree, dec=data["data"]["dec"]*u.degree, frame='icrs').to_string(style="hmsdms", sep="", pad=True, precision=2)[:-1].replace(" ", "")
            self.info.value =  "Querying SDSSDR16 object %s..." % str(objid)
            if objid not in self.hosts_queried.keys():
                self.hosts_queried[objid] = self.get_SDSSDR16_redshift(objid)
                self.hosts_queried[objid]["host_name"] = host_name
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
                    output += "<tr><th><font size='1'>%s: %s</font></th></tr>" % (key, data["data"][key])
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


    def get_SDSSDR16_redshift_spec(self, objid, mode='vot'):
        'get galaxy spectroscopic redshift from SDSS DR16 using SDSS DR16 API'

        SDSS_DR16_url = 'http://skyserver.sdss.org/dr16/SkyServerWS'
        sdss_query = '?cmd=select top 1 p.objid, s.z, s.zerr, s.class, s.zwarning from photoobj as p join specobj as s on s.bestobjid = p.objid where p.objid=%s&format=csv' % objid
        url = '%s/SearchTools/SqlSearch%s' % (SDSS_DR16_url, sdss_query)
        #print(url)
        r = requests.get(url = url, timeout=(5, 10))
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
        sdss_query = '?cmd=select top 1 objid, z, zerr, photoerrorclass from photoz where objid=%s&format=csv' % objid
        url = '%s/SearchTools/SqlSearch%s' % (SDSS_DR16_url, sdss_query)
        #print(url)
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

        results = {}
        photz = self.get_SDSSDR16_redshift_phot(objid, mode='pandas')
        specz = self.get_SDSSDR16_redshift_spec(objid, mode='pandas')
        if not photz is None:
            results["photoz"] = float(photz.z)
            results["photoz_err"] = float(photz.zerr)
        if not specz is None:
            results["specz"] = float(specz.z)
            results["specz_err"] = float(specz.zerr)
        
        return results 

        #params = {
        #    'id': "%s" % objid
        #}
        #r = requests.get(url = "http://skyserver.sdss.org/dr16/en/tools/explore/obj.aspx", params = params)
        #df = pd.read_html(str(r.content))
        #results = {}
        #
        ## extract name and photoz
        #for i in df:
        #    data = i.loc[0][0]
        #    if type(data) is str:
        #        m = re.match("(?P<host>SDSS\sJ.{18})", data[5:])
        #        if m:
        #            results["host_name"] = m["host"]
        #    if data == 'Mjd-Date':
        #        if i.iloc[0][1] == 'photoZ (KD-tree method)':
        #            photozs = i.iloc[1][1].split()
        #            results["photoz"] = photozs[0]
        #            results["photoz_err"] = photozs[2]
        #    if data == "Spectrograph":
        #        results["specz"] = i.iloc[2][1]
        #        results["specz_err"] = i.iloc[3][1]
        #
        ## if specz available query from quicklook
        #if "specz" in results.keys():
        #    r = requests.get(url = "http://skyserver.sdss.org/dr16/en/tools/quicklook/summary.aspx", params = params)
        #    results["specz"] = float(re.findall("Redshift\s\(z\):.*\n.*>(\d.\d+)</td>", r.text)[0])
        #
        #return results

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
        self.plot_stamps(oid=oid)

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
        if non_detections.empty:
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
            remarks = "Early SN candidate%s classified using ALeRCE's stamp classifier (Carrasco-Davis et al. 2021) and the public ZTF stream. Discovery image and light curve in http://alerce.online/object/%s " % (dmdtstr, oid)
            #remarks = "Early SN candidate classified by ALeRCE using the public ZTF stream. Discovery image and light curve in http://alerce.online/object/%s " % (oid)
        else:
            remarks = "SN candidate classified using ALeRCE's stamp classifier (Carrasco-Davis et al. 2021) and the public ZTF stream. Discovery image and light curve in http://alerce.online/object/%s " % oid
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

        # fill all compulsory fielsddev
        try:
            ra_err = max(0.085, float(stats.sigmara)) # use a conservative floor of 85 milliarcsec (see Masci et al. 2019)
            dec_err = max(0.085, float(stats.sigmara)) # use a conservative floor of 85 milliarcsec (see Masci et al. 2019)
        except:
            ra_err = 0.085
            dec_err = 0.085
        report = {
                "ra": {
                    "value": "%s" % RA,
                    "error": "%s" % ra_err,
                    "units": "arcsec"
                    },
                "dec": {
                    "value": "%s" % DEC,
                    "error": "%s" % dec_err,
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

    def get_reset_time(self, response):
        for name in response.headers:
            value = response.headers.get(name)
            if name.endswith('-remaining') and value == '0':
                return int(response.headers.get(name.replace('remaining', 'reset')))
        return None

    def rate_limit_handling(self, response):
        reset = self.get_reset_time(response)
        if (response.status_code == 200):
            if reset != None:
                print("Sleep for " + str(reset + 1) + " sec")
                time.sleep(reset + 1)

    def get_tns(self, api_key, oid):
        'get information about the candidate from TNS'
        
        # get ra, dec
        stats = self.query_objects(oid=oid, format='pandas').drop_duplicates(subset='oid')
        
        url_tns_api="https://www.wis-tns.org/api/get"
        search_url=url_tns_api+'/search'
        search_obj=[("ra", "%f" % stats.meanra), ("dec", "%f" % stats.meandec), ("radius","5"), ("units","arcsec"),
            ("objname",""), ("internal_name","")]
        search_obj=OrderedDict(search_obj)
        search_data=[('api_key',(None, api_key)), ('data',(None,json.dumps(search_obj)))]
        try:
            response=requests.post(search_url, headers=self.tns_headers, files=search_data, timeout=(5, 10))
            self.rate_limit_handling(response)
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
            response = requests.post(json_url, headers=self.tns_headers, files = json_data)
            self.rate_limit_handling(response)
            
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
            response = requests.post(json_url, headers=self.tns_headers, files = json_data)
            self.rate_limit_handling(response)
            
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
        r = self.api('GET', "%s/filters" % url, token)
        idcands = {}
        #print(f'HTTP code: {r.status_code}, {r.reason}')
        #print()
        if r.status_code in (200, 400):
            for i in r.json()["data"]:
                idcands[i["name"]] = i["id"]
        return idcands["ALERCE"]

    # get filter ids
    def get_skyportal_group_id(self, url, token):
        r = self.api('GET', "%s/groups" % url, token)
        idsource = {}
        #print(f'HTTP code: {r.status_code}, {r.reason}')
        #print()
        if r.status_code in (200, 400):    
            for i in r.json()["data"]["user_groups"]:
                idsource[i["nickname"]] = i["id"]
        return idsource["ALERCE"]

    # check if candidate is in SkyPortal
    def isin_skyportal(self, url, token, oid):
        status = self.api('GET', "%s/candidates/%s" % (url, oid), token).json()["status"]
        return status == "success"
    
    
    # Prepare SkyPortal report
    def do_skyportal_report(self, url, token, oid, reporter, verbose=False, test=False):

        # get ALeRCE stats
        if verbose:
            print("Getting stats for object %s" % oid)
        stats = self.query_objects(oid=oid, format='pandas')

        # build report
        report = {}
        report["candidates"] = {}
        report["candidates"]["ra"] = float(stats.meanra)
        report["candidates"]["dec"] = float(stats.meandec)
        report["candidates"]["id"] = oid            # Name of the object
        report["candidates"]["origin"] = "ZTF"        # Origin of the object.
        report["candidates"]["filter_ids"] = [self.get_skyportal_filter_id(url, token)]
        report["annotation"] = {}
        report["annotation"] = {}
        report["annotation"]["obj_type"] = "sn_candidate"
        report["annotation"]["obj_identification"] = "stamp_classifier+visual_inspection"
        if oid in self.candidate_hosts.index:
            if self.candidate_hosts.loc[oid].host_redshift != "NULL" and (report_photoz_skyportal or self.candidate_hosts.loc[oid].host_redshift_spec):
                report["candidates"]["redshift"] = float(self.candidate_hosts.loc[oid].host_redshift)
            if self.candidate_hosts.loc[oid].host_name != "NULL":
                report["annotation"]["host_name"] = self.candidate_hosts.loc[oid].host_name
                report["annotation"]["host_identification"] = "visual_inspection"
                if self.candidate_hosts.loc[oid].host_ra != "NULL" and self.candidate_hosts.loc[oid].host_dec != "NULL":
                    host_ra = float(self.candidate_hosts.loc[oid].host_ra)
                    host_dec = float(self.candidate_hosts.loc[oid].host_dec)
                    report["annotation"]["host_ra"] = host_ra
                    report["annotation"]["host_dec"] = host_dec
                    if self.candidate_hosts.loc[oid].host_offset != "NULL":
                        report["annotation"]["host_offset_arcsec"] = self.candidate_hosts.loc[oid].host_offset # arcseconds
                if self.candidate_hosts.loc[oid].host_source != "NULL":
                    report["annotation"]["host_source"] = self.candidate_hosts.loc[oid].host_source
                if self.candidate_hosts.loc[oid].host_redshift != "NULL":
                    report["annotation"]["host_redshift"] = float(self.candidate_hosts.loc[oid].host_redshift)
                if self.candidate_hosts.loc[oid].host_redshift_error != "NULL":
                    report["annotation"]["host_redshift_error"] = float(self.candidate_hosts.loc[oid].host_redshift_error)
                if self.candidate_hosts.loc[oid].host_redshift_type != "NULL":
                    report["annotation"]["host_redshift_type"] = self.candidate_hosts.loc[oid].host_redshift_type
        report["annotation"]["reporters"] = reporter
        report["annotation"]["alerce_url"] = "https://alerce.online/object/%s" % oid

        return report

    # send to skyportal
    def send_skyportal_report(self, url, token, report):

        data = report["candidates"]
        obj_id = data["id"]
        data["passed_at"] = time.strftime('20%y-%m-%dT%H:%M:%S', time.gmtime())
        
        # check if candidate is not in skyportal
        if self.isin_skyportal(url, token, obj_id):
            print("%s is in skyportal" % obj_id)
        else:
            print("\nReporting candidate")
            # report candidates
            r = self.api('POST', "%s/candidates" % url, token, data=data)
        
            print(f'HTTP code: {r.status_code}, {r.reason}')
            if r.status_code in (200, 400):
                print("Candidate sent")
                display(r.json())
            else:
                print("Unable to post candidate")
    
        # report annotations
        if "annotation" in report.keys():
            print("\nSending annotation")
            data = {}
            data["obj_id"] = obj_id
            data["origin"] = "alerce"
            data["data"] = report["annotation"]
            data["group_ids"] = [self.get_skyportal_group_id(url, token)]
            print()
            
            r = self.api('POST', "%s/annotation" % url, token, data=data)
            
            print(f'HTTP code: {r.status_code}, {r.reason}')
            if r.status_code in (200, 400):
                display(r.json())
                return True
            else:
                print("Unable to post candidate")
                return False
