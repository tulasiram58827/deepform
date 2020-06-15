import json
import requests
import csv
import datetime
import pandas as pd

#create csv 
headers = "file_id,url,file_name,source_service_code,entity_id,callsign,political_file_type,office_type,nielsen_dma_rank,network_affiliation,full_file_name,create_ts,last_update_ts,OCR_needed,doc_angle,OCR_error,FCC_URL,search_term,date"
with open('fcc-2020-all.csv', 'w') as f:
    f.write(headers + '\n')

#setup FCC api
filters = '[{"campaign_year":"2020"},{"source_service_code":"TV"}]'

search_term =  ['order', 'contract', 'invoice', 'receipt']

#loop through search terms
for form in search_term:
    params = {
    ('t', 'opif'),
    ('q', form),
    ('s', 0),
    ('o', 'best'),
    ('f', filters)}
    
    fcc_url = 'https://www.fcc.gov/search/api'
    
    #make initial request to get total number
    r = requests.get(fcc_url,params=params)
    num = r.json()['response']['numFound']
    
    for offset in range(0,num,10):
        params = {
        ('t', 'opif'),
        ('q', form),
        ('s', offset),
        ('o', 'best'),
        ('f', filters)}
        r = requests.get(fcc_url,params=params)
        try:
            records = r.json()['response']['docs']
            for record in records:
                filename = '{fileID}.{ext}'.format(fileID=record['file_id'], ext=record['file_extension'])
                downloadURL = 'https://publicfiles.fcc.gov/api/manager/download/{folder}/{file}.pdf'.format(folder=record['folder_id'], file=record['file_manager_id'])
                OCR_needed = 'NaN' 
                OCR_error = 'NaN'
                page_rotation = 'NaN'
                url = 'NaN'
                current_date = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%Y')

                line = [record['file_id'], url, record['file_name'], 
                    record.get('source_service_code', 'NaN'), record.get('entity_id', 'NaN'), 
                    record.get('callsign', 'NaN'), record.get('political_file_type', 'NaN'), 
                    record.get('office_type', 'NaN'), 
                    record.get('nielsen_dma_rank', 'NaN'), record.get('network_affiliation', 'NaN'), 
                    record.get('full_qualified_file_name', 'NaN'), record.get('create_ts', 'NaN'),
                    record.get('last_update_ts', 'NaN'), OCR_needed, page_rotation, OCR_error, 
                    downloadURL, form, current_date]
                    
                with open('fcc-2020-all.csv','a') as f:
                    wr = csv.writer(f, dialect='excel')
                    wr.writerow(line)
        except ValueError:
            print("Response content is not valid JSON")

df = pd.read_csv('fcc-2020-all.csv')
#remove files that appear in multiple searches and append search terms to indicate
duplicates = df[df['file_id'].duplicated()]
duplicate_files = duplicates['file_id']

for file in duplicate_files:
    rows = df[df['file_id'] == file]
    index = df[df['file_id'] == file].index
    search = rows['search_term'].unique()
    for idx in index:
        df.at[idx,'search_term'] = search
        
df.to_csv('fcc-2020-manifest.csv', index=False)