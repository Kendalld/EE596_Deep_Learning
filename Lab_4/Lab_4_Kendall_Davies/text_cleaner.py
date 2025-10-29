#!/usr/bin/env python3
"""
Text Cleaner for Sherlock Holmes Dataset
This program cleans and formats the sherlock.txt file for better RNN training.
"""

import re
import string

def clean_sherlock_text(input_file, output_file):
    """
    Clean the Sherlock Holmes text file by:
    - Removing page numbers and chapter markers
    - Normalizing whitespace
    - Removing excessive punctuation
    - Keeping only essential characters for text generation
    """
    
    print(f"Reading text from: {input_file}")
    
    # Read the original text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Original text length: {len(text)} characters")
    
    # Step 1: Remove page numbers (common patterns)
    # Remove standalone numbers that are likely page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove numbers at the beginning of lines (often page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Step 2: Remove chapter markers and headers
    # Remove lines that are just numbers or "CHAPTER" followed by numbers
    text = re.sub(r'^\s*CHAPTER\s+\d+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*\d+\.\s*$', '', text, flags=re.MULTILINE)
    
    # Step 3: Clean up excessive whitespace
    # Replace multiple spaces with single space (but preserve single spaces)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace multiple newlines with double newlines (paragraph breaks)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove leading/trailing whitespace from lines (but preserve spaces between words)
    text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE)
    
    # Step 4: Remove or replace problematic characters
    # Keep only printable characters, letters, punctuation, and whitespace
    # This removes control characters and other non-printable characters
    printable_chars = set(string.printable)
    text = ''.join(char for char in text if char in printable_chars)
    
    # Step 5: Normalize punctuation
    # Replace multiple punctuation marks with single ones
    text = re.sub(r'[.]{2,}', '.', text)  # Multiple periods -> single period
    text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamations -> single
    text = re.sub(r'[?]{2,}', '?', text)  # Multiple questions -> single
    
    # Step 6: Clean up quotes and apostrophes
    # Normalize different types of quotes
    text = text.replace('"', '"').replace('"', '"')  # Smart quotes to regular quotes
    text = text.replace(''', "'").replace(''', "'")  # Smart apostrophes to regular
    
    # Step 7: Remove excessive dashes
    text = re.sub(r'-{3,}', '--', text)  # Multiple dashes -> double dash
    
    # Step 8: Final cleanup
    # Remove empty lines at the beginning and end
    text = text.strip()
    
    # Ensure text ends with proper punctuation
    if text and text[-1] not in '.!?':
        text += '.'
    
    print(f"Cleaned text length: {len(text)} characters")
    
    # Write the cleaned text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Cleaned text saved to: {output_file}")
    
    # Show character statistics
    unique_chars = sorted(set(text))
    print(f"\nUnique characters in cleaned text ({len(unique_chars)}):")
    print(''.join(unique_chars))
    
    return text

def analyze_text_quality(text):
    """
    Analyze the quality of the cleaned text
    """
    print("\n=== TEXT QUALITY ANALYSIS ===")
    
    # Character frequency analysis
    char_freq = {}
    for char in text:
        char_freq[char] = char_freq.get(char, 0) + 1
    
    # Sort by frequency
    sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
    
    print("Most frequent characters:")
    for char, freq in sorted_chars[:20]:
        if char == '\n':
            print(f"  '\\n' (newline): {freq}")
        elif char == ' ':
            print(f"  ' ' (space): {freq}")
        else:
            print(f"  '{char}': {freq}")
    
    # Word count
    words = text.split()
    print(f"\nTotal words: {len(words)}")
    print(f"Average word length: {sum(len(word) for word in words) / len(words):.2f}")
    
    # Sentence count
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    print(f"Total sentences: {len(sentences)}")
    print(f"Average sentence length: {sum(len(s.split()) for s in sentences) / len(sentences):.2f} words")

if __name__ == "__main__":
    # File paths
    input_file = "sherlock.txt"
    output_file = "sherlock_cleaned.txt"
    
    try:
        # Clean the text
        cleaned_text = clean_sherlock_text(input_file, output_file)
        
        # Analyze the cleaned text
        analyze_text_quality(cleaned_text)
        
        print(f"\n✅ Text cleaning completed successfully!")
        print(f"📁 Original file: {input_file}")
        print(f"📁 Cleaned file: {output_file}")
        
        # Read original file size for comparison
        with open(input_file, 'r', encoding='utf-8') as f:
            original_size = len(f.read())
        print(f"📊 Original size: {original_size} characters")
        print(f"📊 Cleaned size: {len(cleaned_text)} characters")
        print(f"📊 Characters removed: {original_size - len(cleaned_text)}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}")
        print("Make sure the sherlock.txt file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ Error: {e}")
