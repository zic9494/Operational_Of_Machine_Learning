from dotenv import load_dotenv
import os, requests

class CWA():
    def __init__(self):
        load_dotenv()
        self.__ApiKey = os.getenv("CWA_KEY")
        self.RainData = [None for i in range(6)]
        self.Tags = ['Now', 'Past10Min', 'Past1hr', 'Past3hr', 'Past12hr', 'Past24hr']
    
    def to_dict(self):
        return {
            tag: rain
            for tag, rain in zip(self.Tags, self.RainData)
        }
    
    def get_rain(self, StationId="466920", StationName="臺北"):
        url = 'https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0002-001'
        params = {
            "Authorization": self.__ApiKey,
            "StationId":StationId,
            "StationName":StationName
        }
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            if response.status_code == 401:
                print("API key not work")
            response.raise_for_status()
        
        data = response.json()
        record = data['records']['Station'][0]['RainfallElement']
        for i, v in enumerate(self.Tags):
            self.RainData[i] = float(record[v]['Precipitation'])

if __name__ == "__main__":
    peko = CWA()
    peko.get_rain()