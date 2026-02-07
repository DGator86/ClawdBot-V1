import os
import asyncio
import aiohttp
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Utility to send alerts to Telegram."""
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage" if self.token else None

    async def send_message(self, text: str):
        """Send a message to the configured Telegram chat."""
        if not self.token or not self.chat_id:
            logger.warning("Telegram NOT configured. Skipping alert.")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        resp_text = await response.text()
                        logger.error(f"Telegram API Error: {response.status} - {resp_text}")
                        return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

def send_telegram_alert_sync(text: str):
    """Synchronous wrapper for send_message."""
    notifier = TelegramNotifier()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        asyncio.create_task(notifier.send_message(text))
    else:
        loop.run_until_complete(notifier.send_message(text))
