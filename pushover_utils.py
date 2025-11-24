import http.client
import urllib.parse
import logging
import os
import sys
from typing import Optional
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)

# Load .env from the project (if present)
load_dotenv(find_dotenv())
logger.debug("Loaded .env: PUSHOVER_API_KEY present=%s, PUSHOVER_USER_KEY present=%s",
             bool(os.getenv("PUSHOVER_API_KEY")), bool(os.getenv("PUSHOVER_USER_KEY")))

        
def send_error_notification(error_message: str, context: str = "MyNewHouse Error") -> bool:
    """
    Send error notification via Pushover and log the error
    
    Args:
        error_message: The error message to send
        context: Context/title for the error (e.g., "LinkedIn Posting Error")
        
    Returns:
        bool: True if notification was sent successfully, False otherwise
    """
    # Always log the error first
    logger.error(f"{error_message}")        
#    sys.exit(1)
    
    try:
        # Prepare the message
        message_data = {
            "token": os.getenv('PUSHOVER_API_KEY'),
            "user": os.getenv('PUSHOVER_USER_KEY'),
            "message": f"{context}\n\n{error_message}",
            "title": "MyNewHouse Error",
            "priority": 1,  # High priority for errors
            "sound": "siren"  # Alert sound for errors
        }
        
        # Send the notification
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request("POST", "/1/messages.json",
                    urllib.parse.urlencode(message_data),
                    {"Content-type": "application/x-www-form-urlencoded"})
        
        response = conn.getresponse()
        response_data = response.read().decode()
        
        if response.status == 200:
            logger.info("Pushover error notification sent successfully")
            sys.exit(1)
        else:
            logger.error(f"Failed to send Pushover notification: {response.status} - {response_data}")
            
    except Exception as e:
        logger.error(f"Error sending Pushover notification: {e}")
        sys.exit(1)
    finally:
        try:
            if 'conn' in locals():
                conn.close()
            sys.exit(1) # Always exit after sending error notification
        except Exception:
            logger.error(f"Connection close error")
            sys.exit(1) # Always exit after error

def send_info_notification(info_message: str, context: str = "MyNewHouse Info") -> bool:
    """
    Send informational notification via Pushover and log the info

    Args:
        info_message: The informational message to send
        context: Context/title for the info (e.g., "LinkedIn Posting Success")

    Returns:
        bool: True if notification was sent successfully, False otherwise
    """

    try:
        message_data = {
            "token": os.getenv('PUSHOVER_API_KEY'),
            "user": os.getenv('PUSHOVER_USER_KEY'),
            "message": f"{context}\n\n{info_message}",
            "title": "MyNewHouse Info",
            "priority": 0  # Medium priority for info
#            "sound": None  # No sound for info messages
        }

        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request("POST", "/1/messages.json",
                     urllib.parse.urlencode(message_data),
                     {"Content-type": "application/x-www-form-urlencoded"})

        response = conn.getresponse()
        response_data = response.read().decode()

        if response.status == 200:
            logger.info("Pushover info notification sent successfully")
            return True
        else:
            logger.error(f"Failed to send Pushover info notification: {response.status} - {response_data}")
            return False

    except Exception as e:
        logger.error(f"Error sending Pushover info notification: {e}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            logger.error("Connection close error")

