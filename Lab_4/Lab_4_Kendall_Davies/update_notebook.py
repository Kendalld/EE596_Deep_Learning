#!/usr/bin/env python3
"""
Update Lab 4 notebook to use cleaned text
This script modifies the notebook to use sherlock_cleaned.txt instead of sherlock.txt
"""

import json
import re

def update_notebook_to_use_cleaned_text(notebook_path):
    """
    Update the notebook to use the cleaned text file
    """
    
    print(f"Reading notebook: {notebook_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and update the cell that loads sherlock.txt
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Look for the line that loads sherlock.txt
            if "sherlock.txt" in source:
                print(f"Found sherlock.txt reference in cell {i}")
                
                # Replace sherlock.txt with sherlock_cleaned.txt
                new_source = []
                for line in cell['source']:
                    if 'sherlock.txt' in line:
                        new_line = line.replace('sherlock.txt', 'sherlock_cleaned.txt')
                        print(f"  Updated: {line.strip()} -> {new_line.strip()}")
                        new_source.append(new_line)
                    else:
                        new_source.append(line)
                
                cell['source'] = new_source
                break
    
    # Write the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Notebook updated successfully!")

if __name__ == "__main__":
    notebook_path = "Lab_4_Kendall_Davies.ipynb"
    
    try:
        update_notebook_to_use_cleaned_text(notebook_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {notebook_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
