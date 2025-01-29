import logging
from fastapi import HTTPException


logger = logging.getLogger(__name__)

class CUSTOM_HTTPEXCEPTION(HTTPException):
    def __init__(self, message, status_code):
        # Note the corrected order of parameters here
        super().__init__(status_code=status_code, detail=message)
        logger.error(f"{status_code} - {message}")

class NOT_FOUND_HTTPEXCEPTION(CUSTOM_HTTPEXCEPTION):
    def __init__(self, message):
        super().__init__(message, 404)

class NOT_AUTHORIZED_HTTPEXCEPTION(CUSTOM_HTTPEXCEPTION):
    def __init__(self, message):
        super().__init__(message, 401)

class BAD_REQUEST_HTTPEXCEPTION(CUSTOM_HTTPEXCEPTION):
    def __init__(self, message):
        super().__init__(message, 400)

class FORBIDDEN_HTTPEXCEPTION(CUSTOM_HTTPEXCEPTION):
    def __init__(self, message):
        super().__init__(message, 403)

class CONFLICT_HTTPEXCEPTION(CUSTOM_HTTPEXCEPTION):
    def __init__(self, message):
        super().__init__(message, 409)

class PAYLOAD_TOO_LARGE_HTTPEXCEPTION(CUSTOM_HTTPEXCEPTION):
    def __init__(self, message):
        super().__init__(message, 413)

class INTERNAL_SERVER_ERROR_HTTPEXCEPTION(CUSTOM_HTTPEXCEPTION):
    def __init__(self, message):
        super().__init__(message, 500)

