"""
SMS reminder service.
Primary:  Twilio (international)
Fallback: Fast2SMS (India, cheaper)
"""
import logging
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def send_sms(to: str, message: str) -> bool:
    """Send an SMS. Returns True on success."""
    if settings.TWILIO_ACCOUNT_SID:
        return await _send_twilio(to, message)
    if settings.FAST2SMS_API_KEY:
        return await _send_fast2sms(to, message)
    logger.warning("No SMS provider configured — message not sent: %s", message)
    return False


async def _send_twilio(to: str, message: str) -> bool:
    try:
        from twilio.rest import Client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        client.messages.create(body=message, from_=settings.TWILIO_FROM_NUMBER, to=to)
        logger.info("Twilio SMS sent to %s", to)
        return True
    except Exception as e:
        logger.error("Twilio error: %s", e)
        return False


async def _send_fast2sms(to: str, message: str) -> bool:
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://www.fast2sms.com/dev/bulkV2",
                headers={"authorization": settings.FAST2SMS_API_KEY},
                json={"route": "q", "message": message, "numbers": to},
            )
            r.raise_for_status()
        logger.info("Fast2SMS sent to %s", to)
        return True
    except Exception as e:
        logger.error("Fast2SMS error: %s", e)
        return False
