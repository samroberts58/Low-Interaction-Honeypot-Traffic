# -*- coding: utf-8 -*-
"""
Data Capstone Project:  Low-Interaction Honeypot Traffic: Modeling the Search
Data:                   Rapid7 Heisenberg Cloud Honeypot cowrie Logs

Data Preprocessing - Convert JSON files to structured CSV
"""

import json
import codecs
import csv
import os

# Load variable with current working directory to loop through files.
directory = os.getcwd()

for i in os.listdir(directory):  
    print(i)
    data = []
    with codecs.open(i,'rU','utf-8') as sourceFile:
        for line in sourceFile:
            data.append(json.loads(line))

    # Identify all categories/headers within 'data'
    columns = set()
    for a in data:
        for j in a:
            header = j
            columns.add(header)
    
    columns = list(columns)

    
    # Write the JSON to a CSV 
    count = 0
    outputFile = open(i[:5]+'.csv','w', newline='')
    for b in data:
        if count == 0:
            for k in columns:
                outputFile.write(k)
                if k != columns[-1]:
                    outputFile.write(",")
                else:
                    outputFile.write("\n")
        count += 1
        for k in columns:
            if k in b.keys():
                outputFile.write('"' + b[k] + '"')
            if k != columns[-1]:
                outputFile.write(",")
            else:
                outputFile.write("\n")
    
    outputFile.close()
    
    # Pull out unique Source IP's for Geographical matching
    src_ip = set()
    for c in data:
        for n,v in c.items():
            src_ip_item = c['src_ip']
            src_ip.add(src_ip_item)
    
    src_ip = list(src_ip)

    # Write the unique src_ip list to a CSV    
    outputFile = open(i[:5]+'_src_ip.csv','w', newline='')
    outputWriter = csv.writer(outputFile)
    for d in src_ip:
        outputWriter.writerow(d.split())
    outputFile.close()

