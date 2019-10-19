import requests     # 2.19.1
import json         # 2.0.9
import pandas as pd # 0.23.4
import pprint


def map_jobs(jobs):
    if openfigi_apikey:
        openfigi_headers['X-OPENFIGI-APIKEY'] = openfigi_apikey
    response = requests.post(url=openfigi_url, headers=openfigi_headers,
                             json=jobs)
    if response.status_code != 200:
        raise Exception('Bad response code {}'.format(str(response.status_code)))
    return response.json()


jobs = [
    {'idType': 'TICKER', 'idValue': 'ADS', 'micCode': 'XETR'},
    {'idType': 'TICKER', 'idValue': 'BAS', 'micCode': 'XETR'},
    {'idType': 'TICKER', 'idValue': 'DTE', 'micCode': 'XETR'},
    {'idType': 'TICKER', 'idValue': 'SAP', 'micCode': 'XETR'},
    {'idType': 'TICKER', 'idValue': 'SIE', 'micCode': 'XETR'}
]

openfigi_apikey = 'xxx'  # Please put your own API Key here
openfigi_url = 'https://api.openfigi.com/v2/mapping'


openfigi_headers = {'Content-Type': 'text/json'}

job_results = map_jobs(jobs)
#pprint.pprint(job_results)

just_dictionaries = [d['data'][0] for d in job_results]
df_figi = pd.DataFrame.from_dict(just_dictionaries)
print(df_figi)

