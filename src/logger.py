import logging 
import os
from datetime import datetime

#The log file paths are named by the time they were created
#The log files are stored in directory indicating their date
#Multiple files can be stored in the folder for a particular day
# Hence logs_path (directory) and LOG_FILE_PATH (actual file insde directory)


LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

#Now that the direcotry are ceated and the file path determined
#The following code will determine the format of the message

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

#Testing logger
# if __name__=="__main__":
#     logging.info('Logging has started')