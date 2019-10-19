import os
import requests     # 2.19.1
import json         # 2.0.9
import pandas as pd # 0.23.4
import pprint
import pickle

class perm_id_reader():


    permid_ref_value_map = {}
    permid_ref_value_map_path = None
    destination_path = None
    request_url = "https://api.thomsonreuters.com/permid/match"  # API endpoint
    tickers = None

    headers = {
        'Content-Type': 'text/plain',
        'Accept': 'application/json',
        'x-ag-access-token': 'xxx', # use yours
        'x-openmatch-numberOfMatchesPerRecord': '1',  # only return 1 match per ticker
        'x-openmatch-dataType': 'Organization'  # only match to "Organization", not "Person" or "Instrument"
    }

    def __init__(self, ticker_map, ref_pickle_path=None, destination_path=None ):

        self.tickers = ticker_map

        if ref_pickle_path is not None:
            self.permid_ref_value_map_path = ref_pickle_path

        if self.permid_ref_value_map_path is not None and os.path.exists(self.permid_ref_value_map_path):
            print('found reference cache ' + self.permid_ref_value_map_path)
            fileObject = open(self.permid_ref_value_map_path, 'rb')
            ref_map_from_pickle = pickle.load(fileObject)
            self.permid_ref_value_map = ref_map_from_pickle
        if destination_path is not None:
            self.destination_path = destination_path


    def permid_data(self, permid_url):
        permid_headers = {
            'Accept': 'text/turtle',
        }

        permid_params = {
            'format': 'json-ld',
            'access-token': 'xxx'
        }

        # The actual request
        permid_response = requests.get(permid_url, headers=self.headers, params=permid_params)

        # Convert the response to JSON
        permid_data = json.loads(permid_response.content)

        return permid_data



    def get_stored_dat(self, permid_url):

        if 'https://permid.org' not in permid_url:
            return  permid_url

        if  permid_url not in self.permid_ref_value_map:
            data = self.permid_data(permid_url)
            self.permid_ref_value_map[permid_url] = data['prefLabel']

        return self.permid_ref_value_map[permid_url]


    def run_query(self):
        '''
        This will get all permid urls.
        for links which refer to geonames, simply download https://download.geonames.org/export/dump/countryInfo.txt
        and extend this class to replace the geonames links with an entry from the data in the link.

        :return:
        '''

        text_field = 'Standard Identifier\n'

        # For every ticker, we will add a new line and specify the Market Identifier Code (MIC) / Exchange

        for ticker in self.tickers.keys():
            identifier = 'TICKER:' + ticker + '&&MIC:' + tickers[ticker] + '\n'
            text_field += identifier


        response = requests.post(self.request_url, headers=self.headers, data=text_field)
        r = response.json()
        pprint.pprint(r['outputContentResponse'])

        print('retrieving details ...')

        # Create an empty dictionary
        permid_dict = {}

        # Loop through all tickers and put the data in the dictionary
        for ticker, i in zip(self.tickers, r['outputContentResponse']):
            # The PermID url for the ticker from the response earlier
            permid_url = i['Match OpenPermID']

            # Use the function defined above to download the data
            data = self.permid_data(permid_url)
            #print(data)

            # Put the desired data in a dictionary for the ticker
            permid_dict[ticker] = {
                'company': data['vcard:organization-name'],
                #'IPO': data['hasIPODate'],
                #'Primary_business_sector': self.get_stored_dat(data['hasPrimaryBusinessSector']),
                'Primary_economic_sector': self.get_stored_dat(data['hasPrimaryEconomicSector']),
                'Primary_industry_group': self.get_stored_dat(data['hasPrimaryIndustryGroup']),
                #'Incorporated_in': self.get_stored_dat(data['isIncorporatedIn'])
                'Domiciled_in': self.get_stored_dat(data['isDomiciledIn']),
                #'phone': data['tr-org:hasHeadquartersPhoneNumber'],
                #'LEI': data['tr-org:hasLEI'],
                #'permid': data['tr-common:hasPermId'],
                #'address': data['mdaas:HeadquartersAddress'],
                'permid_url': permid_url
            }


        df_permid = pd.DataFrame.from_dict(permid_dict,
                                               orient='index')  # Orient='index' for data in rows instead of columns


        if self.destination_path is not None:
            print('writing result ' +self.destination_path)
            df_permid.to_csv(self.destination_path)

        if self.permid_ref_value_map_path is not None:
            print('saving ref cache ' + self.permid_ref_value_map_path)
            fileObject = open(self.permid_ref_value_map_path, 'wb')
            pickle.dump(self.permid_ref_value_map, fileObject)
            fileObject.close()

        return df_permid

if __name__ == "__main__":

    tickers = {'AAPL': 'XNGS', 'ADS': 'XETR', 'BAS': 'XETR', 'DTE': 'XETR', 'SAP': 'XETR',
               'SIE': 'XETR'}  # Apple, Adidas, BASF, Deutsche Telekom, SAP & Siemens

    tickers_short = {'AAPL': 'XNGS', 'ADS': 'XETR'}
    reader = perm_id_reader(tickers, r'C:\data\ref_data_map.pkl', r'C:\data\result_permid.csv')
    df = reader.run_query()
    print(df)