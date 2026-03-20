"""
Common utility functions
"""

import os
import json
from typing import Any, Dict, List


def read_json(file_path: str) -> Any:
    """
    Read JSON file
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Parsed JSON content
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, file_path: str, indent: int = 4) -> None:
    """
    Write data to JSON file
    
    Args:
        data: Data to write
        file_path: Output file path
        indent: JSON indentation
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w+", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists
    
    Args:
        path: Directory path
    
    Returns:
        The same path
    """
    os.makedirs(path, exist_ok=True)
    return path
