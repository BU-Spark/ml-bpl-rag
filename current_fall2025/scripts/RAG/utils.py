#!/usr/bin/env python3
"""
Utility functions for RAG system.
Currently unused but preserved for potential future use.
"""

import logging
import requests
from typing import Optional, Dict


def safe_get_json(url: str) -> Optional[Dict]:
    """
    Safely fetch JSON from a URL with error handling.
    
    Args:
        url: URL to fetch JSON from
        
    Returns:
        JSON dict if successful, None otherwise
    """
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None


def extract_text_from_json(json_data: Dict) -> str:
    """
    Extract text from JSON data structure.
    
    Args:
        json_data: JSON dict with nested structure
        
    Returns:
        Concatenated text from specified fields
    """
    if not json_data:
        return ""
    fields = [
        "title_info_primary_tsi",
        "abstract_tsi",
        "subject_geographic_sim",
        "genre_basic_ssim",
        "genre_specific_ssim",
        "date_tsim",
    ]
    parts = []
    for field in fields:
        if field in json_data.get("data", {}).get("attributes", {}):
            val = json_data["data"]["attributes"][field]
            if val:
                parts.append(str(val))
    return " ".join(parts) if parts else "No content available"

