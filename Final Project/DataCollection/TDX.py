from dotenv import load_dotenv
import os, requests, json, logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
class Park():
    def __init__(self):
        self.ParkId = None
        self.TotalSpaces = None
        self.AvailableSpaces = None
        self.FullStatus = None
        self.time = None
    
    def to_dict(self):
        return {
            "ParkId": self.ParkId,
            "TotalSpaces": self.TotalSpaces,
            "AvailableSpaces": self.AvailableSpaces,
            "FullStatus": self.FullStatus,
            "time": self.time.isoformat() if self.time else None,
        }

class TDX():
    def __init__(self):
        load_dotenv(Path(__file__).parent / '.env')
        self.__ApiId = os.getenv("TDX_ID")
        self.__ApiKey = os.getenv("TDX_KEY")
        self.token = None
        self.auth()
        self.response = None
        self.parks = []
    
    def to_dict(self):
        return [park.to_dict() for park in self.parks]

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
        try:
            logger.info("TDX auth request start")
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("TDX auth request failed")
            raise
        
        token_data = response.json()
        self.token = token_data['access_token']
        logger.info("TDX auth request finished")

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
            try:
                logger.info("TDX data request start")
                response = requests.get(url, params=params, headers=header)
                
                if response.status_code == 401:
                    logger.warning("TDX token expired, refreshing token")
                    self.auth()
                    header["authorization"] = "Bearer " + self.token
                    response = requests.get(url, params=params, headers=header)

                self.response = response.json()
                response.raise_for_status()
                logger.info("TDX data request end")
            except requests.RequestException:
                logger.exception("TDX parking request failed")
                raise
                
        else:
            file_path = Path(__file__).parent / 'response_1779784573186.json'
            with open(file_path, 'r', encoding='utf-8') as file:
                self.response = json.load(file)
        
        self.handelParkData()

    def handelParkData(self):
        datas = self.response['ParkingAvailabilities']
        for i in datas:
            temp = Park()
            temp.ParkId = i['CarParkID']
            temp.TotalSpaces = int(i['TotalSpaces'])
            temp.AvailableSpaces = int(i['AvailableSpaces'])
            temp.FullStatus = int(i['FullStatus'])
            temp.time = datetime.fromisoformat(i['DataCollectTime'])
            self.parks.append(temp)
        
            
        
    

if __name__ == "__main__":
    peko = TDX()
    peko.getParkSpace(test=False)
    for i in peko.parks:
        print(i.ParkId)
