#!/usr/bin/env python3

import nltk
import os

def setup_nltk():
    """Download required NLTK data packages"""
    print("Setting up NLTK data...")
    
    # Create custom data directory if needed
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    # Download required packages
    packages = [
        'punkt',           # for tokenization
        'stopwords',       # for filtering stopwords
        'averaged_perceptron_tagger'  # for POS tagging
    ]
    
    for package in packages:
        print(f"Downloading {package}...")
        nltk.download(package, download_dir=nltk_data_dir, quiet=False)
    
    print("NLTK setup complete!")

if __name__ == "__main__":
    setup_nltk()