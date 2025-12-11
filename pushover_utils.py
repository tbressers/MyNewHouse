import http.client
import urllib.parse
import logging
import os
import re
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)

# Load .env from the project (if present)
load_dotenv(find_dotenv())
logger.debug("Loaded .env: PUSHOVER_API_KEY present=%s, PUSHOVER_USER_KEY present=%s",
             bool(os.getenv("PUSHOVER_API_KEY")), bool(os.getenv("PUSHOVER_USER_KEY")))

# Global flag for dry-run mode
_dry_run_mode = False

def set_dry_run_mode(enabled: bool):
    """Set the dry-run mode for all Pushover notifications"""
    global _dry_run_mode
    _dry_run_mode = enabled
    if enabled:
        logger.info("Pushover notifications: DRY-RUN mode enabled")

def _split_message(message: str, max_length: int = 512, max_chunks: int = 5) -> list[str]:
    """
    Split message into chunks of max_length, preferring to split after URLs.
    Maximum of max_chunks messages will be created; text beyond that is truncated.
    
    Args:
        message: The message to split
        max_length: Maximum length per chunk (default 512 for Pushover)
        max_chunks: Maximum number of chunks to create (default 5)
        
    Returns:
        list: List of message chunks (maximum max_chunks items)
    """
    if len(message) <= max_length:
        return [message]
    
    chunks = []
    remaining = message
    url_pattern = r'https?://\S+'
    
    while remaining and len(chunks) < max_chunks:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break
        
        chunk = remaining[:max_length]
        
        # Find all URLs in the entire remaining text (not just chunk)
        urls_in_remaining = list(re.finditer(url_pattern, remaining))
        
        # Find URLs that start before max_length
        urls_before_limit = [url for url in urls_in_remaining if url.start() < max_length]
        
        if urls_before_limit:
            # Use the last URL that starts before the limit, but include its full length
            last_url = urls_before_limit[-1]
            split_pos = last_url.end()
            
            # Only split here if it doesn't exceed max_length by too much (allow 50 chars for URL overage)
            if split_pos <= max_length + 50:
                chunks.append(remaining[:split_pos])
                remaining = remaining[split_pos:].lstrip()
            else:
                # URL is too long, split at max_length at word boundary
                split_pos = max_length
                last_space = chunk.rfind(' ')
                if last_space > max_length * 0.7:
                    split_pos = last_space + 1
                chunks.append(remaining[:split_pos])
                remaining = remaining[split_pos:].lstrip()
        else:
            # No URL found, split at max_length at word boundary if possible
            split_pos = max_length
            last_space = chunk.rfind(' ')
            if last_space > max_length * 0.7:
                split_pos = last_space + 1
            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos:].lstrip()
    
    return chunks

def send_error_notification(error_message: str, context: str = "MyNewHouse Error") -> bool:
    """
    Send error notification via Pushover and log the error
    
    Args:
        error_message: The error message to send
        context: Context/title for the error (e.g., "LinkedIn Posting Error")
        
    Returns:
        bool: True if notification was sent successfully, False otherwise
    """    
    if _dry_run_mode:
        logger.info(f"DRY-RUN: Would send error notification: {context}\n{error_message}")
        return True
    else:
        logger.info(f"{context}\n{error_message}")
    
    try:
        # Split message if needed
        full_message = f"{context}\n\n{error_message}"
        message_chunks = _split_message(full_message)
        
        all_sent = True
        for i, chunk in enumerate(message_chunks):
            # Prepare the message
            message_data = {
                "token": os.getenv('PUSHOVER_API_KEY'),
                "user": os.getenv('PUSHOVER_USER_KEY'),
                "message": chunk,
                "title": "MyNewHouse Error" + (f" (Part {i+1}/{len(message_chunks)})" if len(message_chunks) > 1 else ""),
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
            conn.close()
            
            if response.status == 200:
                logger.info(f"Pushover error notification sent successfully (part {i+1}/{len(message_chunks)})")
            else:
                logger.error(f"Failed to send Pushover notification: {response.status} - {response_data}")
                all_sent = False
                
        return all_sent
            
    except Exception as e:
        logger.error(f"Error sending Pushover notification: {e}")
        return False
    finally:
        try:
            if 'conn' in locals():
                conn.close()
        except Exception:
            logger.error("Connection close error")

def send_info_notification(info_message: str, context: str = "MyNewHouse Info") -> bool:
    """
    Send informational notification via Pushover and log the info

    Args:
        info_message: The informational message to send
        context: Context/title for the info (e.g., "LinkedIn Posting Success")

    Returns:
        bool: True if notification was sent successfully, False otherwise
    """
    logger.info(f"{context}: {info_message}")  # Log the info message

    if _dry_run_mode:
        logger.info(f"DRY-RUN: Would send info notification: {context}\n{info_message}")
        return True

    try:
        # Split message if needed
        full_message = f"{context}\n\n{info_message}"
        message_chunks = _split_message(full_message)
        
        all_sent = True
        for i, chunk in enumerate(message_chunks):
            message_data = {
                "token": os.getenv('PUSHOVER_API_KEY'),
                "user": os.getenv('PUSHOVER_USER_KEY'),
                "message": chunk,
                "title": "MyNewHouse Info" + (f" (Part {i+1}/{len(message_chunks)})" if len(message_chunks) > 1 else ""),
                "priority": 0  # Medium priority for info
            }

            conn = http.client.HTTPSConnection("api.pushover.net:443")
            conn.request("POST", "/1/messages.json",
                         urllib.parse.urlencode(message_data),
                         {"Content-type": "application/x-www-form-urlencoded"})

            response = conn.getresponse()
            response_data = response.read().decode()

            if response.status == 200:
                logger.info(f"Pushover info notification sent successfully (part {i+1}/{len(message_chunks)})")
            else:
                logger.error(f"Failed to send Pushover info notification: {response.status} - {response_data}")
                all_sent = False

        return all_sent

    except Exception as e:
        logger.error(f"Error sending Pushover info notification: {e}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            logger.error("Connection close error")

