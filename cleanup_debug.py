#!/usr/bin/env python3
"""
Cleanup script to remove all debug print statements and excessive logging
from the insurance claim AI codebase.
"""

import re
import os
from pathlib import Path

def clean_file(filepath):
    """Remove debug print statements from a file."""
    print(f"Processing: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_lines = content.count('\n')
    
    # Remove print statements with debug tags like [BILLING], [REQ], [SSE], etc.
    content = re.sub(r'^\s*print\([^)]*\[(BILLING|REQ|REQUIREMENTS|SSE|ORCHESTRATOR|APP|PDF|IMAGE)\][^)]*\)\n', '', content, flags=re.MULTILINE)
    
    # Remove print statements with separator lines (=== or ---)
    content = re.sub(r'^\s*print\(["\']={20,}["\'][^)]*\)\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*print\(["\']={20,}.*?\\n["\'][^)]*\)\n', '', content, flags=re.MULTILINE)
    
    # Remove excessive logger.info statements with debug tags
    content = re.sub(r'^\s*logger\.info\(f?["\'].*?\[(BILLING|REQ|REQUIREMENTS|DEBUG)\].*?["\'][^)]*\)\n', '', content, flags=re.MULTILINE)
    
    # Remove empty print() statements
    content = re.sub(r'^\s*print\(\)\n', '', content, flags=re.MULTILINE)
    
    # Remove multi-line print statements for debug output
    content = re.sub(r'^\s*print\(f?["\'].*?Text preview.*?["\'][^)]*\)\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*print\(.*?repr\(.*?\)\)\n', '', content, flags=re.MULTILINE)
    
    new_lines = content.count('\n')
    removed = original_lines - new_lines
    
    if removed > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Removed {removed} debug lines")
        return removed
    else:
        print(f"  - No debug statements found")
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
    print(f"Total debug lines removed: {total_removed}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
