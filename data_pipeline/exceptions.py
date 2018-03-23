"""
File to declare specific excpetions that may occur in the pipeline
"""

class InvalidDataTypeException(Exception):
    """Error when a data provided cannot be transformed in a DataPipe
    """
    def __init__(self, base_exception, message):
        super(InvalidDataTypeException, self).__init__(base_exception, message)