import os


def resolve_project_root(current_file: str) -> str:
    """
    Resolve the project root directory from the current file path.
    
    Args:
        current_file: Path to the current file
        
    Returns:
        Absolute path to the project root directory
    """
    here = os.path.dirname(os.path.abspath(current_file))
    return os.path.abspath(os.path.join(here, os.pardir))
