from datetime import datetime
from pathlib import Path
import csv

class Holiday:
    def __init__(self):
        pass

    def isHoliday(self, Today: datetime):
        file_path = Path(__file__).parent / '115年中華民國政府行政機關辦公日曆表.csv'
        with  open(file_path, 'r') as file:
            HolidayDate = csv.reader(file)
        
            for i, row_date in enumerate(HolidayDate):
                if i == 0:
                    continue
                Date_obj = datetime.strptime(row_date[0], "%Y%m%d").date()
                if Today.date() == Date_obj:
                    match int(row_date[2]) :    
                        case 0:
                            return 0
                        case 2:
                            return 1

if __name__ == "__main__":
    peko = Holiday()
    print(peko.isHoliday(datetime.today()))