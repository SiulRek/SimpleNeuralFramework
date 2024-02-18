import functools
from rich.console import Console
from rich.traceback import Traceback


def rich_exception_handler(exc_type, exc_value, exc_traceback):
    """ 
    A rich exception handler that prints the traceback in a more readable format.
    
    Usage:
    ```
    import sys
    sys.excepthook = rich_exception_handler
    
    # Your code here
    ```
    """
    
    console = Console()
    traceback = Traceback.from_exception(exc_type, exc_value, exc_traceback)
    console.print(traceback)


def rich_test_decorator(raise_exception=True):
    """ 
    A rich test decorator that prints the traceback in a more readable format.

    Args:
        raise_exception (bool): Whether to raise the exception after printing the traceback or not.
    """
    def actual_decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            try:
                test_func(*args, **kwargs)
            except Exception as e:
                console = Console()
                console.print(Traceback.from_exception(type(e), e, e.__traceback__))
                if raise_exception:
                    raise e
        return wrapper
    return actual_decorator


