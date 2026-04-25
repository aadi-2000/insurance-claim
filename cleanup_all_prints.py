#!/usr/bin/env python3
"""
Comprehensive cleanup script to remove ALL print statements from src directory.
"""

import re
import os
from pathlib import Path

def clean_file(filepath):
    """Remove ALL print statements from a file."""
    print(f"Processing: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_count = len(lines)
    cleaned_lines = []
    i = 0
    removed_count = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if line contains a print statement
        if re.match(r'^\s*print\(', line):
            # Handle multi-line print statements
            if '(' in line and ')' not in line:
                # Multi-line print - skip until closing parenthesis
                removed_count += 1
                i += 1
                while i < len(lines) and ')' not in lines[i]:
                    removed_count += 1
                    i += 1
                if i < len(lines):
                    removed_count += 1
                    i += 1
                continue
            else:
                # Single-line print
                removed_count += 1
                i += 1
                continue
        
        # Keep the line
        cleaned_lines.append(line)
        i += 1
    
    if removed_count > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        print(f"  ✓ Removed {removed_count} print statements")
        return removed_count
    else:
        print(f"  - No print statements found")
        return 0

def main():
    """Clean all Python files in src directory."""
    src_dir = Path('src')
    
    if not src_dir.exists():
        print("Error: src directory not found")
        return
    
    total_removed = 0
    files_cleaned = 0
    
    # Find all Python files in src directory
    python_files = list(src_dir.rglob('*.py'))
    
    print(f"Found {len(python_files)} Python files to process\n")
    
    for filepath in python_files:
        removed = clean_file(filepath)
        if removed > 0:
            files_cleaned += 1
            total_removed += removed
    
    print(f"\n{'='*60}")
    print(f"Cleanup Complete!")
    print(f"Files cleaned: {files_cleaned}/{len(python_files)}")
    print(f"Total print statements removed: {total_removed}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
