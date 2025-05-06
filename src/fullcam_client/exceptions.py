"""
Custom exceptions for the FullCAM client
"""


class FullCAMClientError(Exception):
    """Base exception for all FullCAM client errors"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class FullCAMAPIError(FullCAMClientError):
    """Exception raised for errors returned by the FullCAM API"""

    def __init__(
        self, message: str, status_code: int | None = None, response_text: str | None = None
    ):
        self.status_code = status_code
        self.response_text = response_text
        error_msg = message
        if status_code:
            error_msg += f" (Status code: {status_code})"
        super().__init__(error_msg)
