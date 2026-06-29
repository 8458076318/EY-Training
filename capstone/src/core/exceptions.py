"""Domain exceptions mapped to HTTP responses in the API layer."""


class AppException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AppException):
    def __init__(self, resource: str):
        super().__init__(f"{resource} not found", status_code=404)


class AuthError(AppException):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class PermissionError(AppException):
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, status_code=403)


class AgentError(AppException):
    def __init__(self, agent: str, message: str):
        super().__init__(f"Agent '{agent}' failed: {message}", status_code=502)


class LLMFallbackExhausted(AgentError):
    def __init__(self):
        super().__init__("day_planner", "All LLM providers exhausted")
