from dotenv import load_dotenv
import os, requests, logging

logger = logging.getLogger(__name__)
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
        
        logger.info("CWA request start")
        try:
            response = requests.get(url, params=params, timeout=10)
        except requests.RequestException:
            logger.exception("CWA network request failed")
            raise

        if response.status_code != 200:
            logger.error("CWA request failed | status_code=%s", response.status_code)
            response.raise_for_status()
        
        data = response.json()
        record = data['records']['Station'][0]['RainfallElement']
        for i, v in enumerate(self.Tags):
            self.RainData[i] = float(record[v]['Precipitation'])
        logger.info("CWA request finished")

if __name__ == "__main__":
    peko = CWA()
    peko.get_rain()