from dotenv import load_dotenv
import os, requests, json
from pathlib import Path

class TDX():
    def __init__(self):
        load_dotenv(Path(__file__).parent / '.env')
        self.__ApiId = os.getenv("TDX_ID")
        self.__ApiKey = os.getenv("TDX_KEY")
        self.token = None
        self.auth()

    def auth(self):
        url = 'https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token'
        headers = {
            'content-type' : 'application/x-www-form-urlencoded'
        }
        data = {
            'grant_type' : 'client_credentials',
            'client_id' : self.__ApiId,
            'client_secret' : self.__ApiKey
        }
        response = requests.post(url, headers=headers, data=data).json()
        self.token = response['access_token']


    def getParkSpace(self, City="Taipei",test=False):
        url = 'https://tdx.transportdata.tw/api/basic/v1/Parking/OffStreet/ParkingAvailability/City/' + City
        params = {
            'top':1000,
            'format':'json'
        }
        header = {
            'authorization': 'Bearer ' + self.token,
            'Accept-Encoding': 'gzip'
        }
        if not test:   
            self.auth()
            response = requests.get(url, params=params, headers=header).json()
        else:
            file_path = Path(__file__).parent / 'response_1779770296253.json'
            with open(file_path, 'r', encoding='utf-8') as file:
                response = json.load(file)


        
            
        
    

if __name__ == "__main__":
    peko = TDX()
    peko.getParkSpace(test=True)
