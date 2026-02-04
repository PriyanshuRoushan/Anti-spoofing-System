from fastapi import HTTPException

class APIException(HTTPException):
    """Base exception for general configuration errors."""
    
    
    def __init__(
        self, 
        status_code: int = 500, 
        detail: str = "An error occurred",
        exceptionType: str = "APIException"
    ):
        super().__init__(
            status_code=status_code, 
            detail= { 
                     
                "status": "error",
                "message": detail, 
                "type" : exceptionType,
                "path" : "/api/voice-detection"
                
                }
            )
        
