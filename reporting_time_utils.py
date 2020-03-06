from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import re

class tns_scraping(object):
    
    def __init__(self):
        self.driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
        
    def get_reporting_date(self, TNS_name):
        self.driver.get("https://wis-tns.weizmann.ac.il/object/"+str(TNS_name)+"/discovery-cert")
        content = self.driver.page_source
        soup = BeautifulSoup(content)
        text = soup.get_text()
        index = text.find('Date Received')
        size = 40
        full_date = soup.get_text()[index:index+size]
        date, time = full_date.split(" ")[-2:]
        return date, time

if __name__ == "__main__":
    

























