#!/bin/bash
#
# Pre-commit hook to prevent committing files larger than a specified size limit.
# When files exceed the limit, they are automatically added to .gitignore and
# documented in README.md.
#
# Configuration:
#   MAX_FILE_SIZE_MB: Maximum file size in MB (default: 50)
#   README_FILE: Path to README file (default: README.md)
#   GITIGNORE_FILE: Path to .gitignore file (default: .gitignore)
#

set -e

# Configuration with defaults
MAX_FILE_SIZE_MB="${MAX_FILE_SIZE_MB:-50}"
README_FILE="${README_FILE:-README.md}"
GITIGNORE_FILE="${GITIGNORE_FILE:-.gitignore}"
MAX_FILE_SIZE_BYTES=$((MAX_FILE_SIZE_MB * 1024 * 1024))

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Track if we made any changes
CHANGES_MADE=false
LARGE_FILES=()

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

echo "Checking staged files for size limit (${MAX_FILE_SIZE_MB}MB)..."

# Check each staged file
for file in $STAGED_FILES; do
    # Skip if file doesn't exist (might have been deleted)
    if [ ! -f "$file" ]; then
        continue
    fi
    
    # Get file size
    FILE_SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
    
    if [ "$FILE_SIZE" -gt "$MAX_FILE_SIZE_BYTES" ]; then
        FILE_SIZE_MB=$(awk "BEGIN {printf \"%.2f\", $FILE_SIZE / 1024 / 1024}")
        echo -e "${RED}WARNING:${NC} File '$file' is ${FILE_SIZE_MB}MB (exceeds ${MAX_FILE_SIZE_MB}MB limit)"
        LARGE_FILES+=("$file")
    fi
done

# If no large files found, exit successfully
if [ ${#LARGE_FILES[@]} -eq 0 ]; then
    exit 0
fi

# Process large files
echo -e "\n${YELLOW}Processing large files...${NC}"

for file in "${LARGE_FILES[@]}"; do
    # Unstage the file
    git reset HEAD "$file" 2>/dev/null || true
    echo "  Unstaged: $file"
    
    # Add to .gitignore if not already present
    if [ -f "$GITIGNORE_FILE" ]; then
        if ! grep -Fxq "$file" "$GITIGNORE_FILE"; then
            echo "$file" >> "$GITIGNORE_FILE"
            echo "  Added to .gitignore: $file"
            CHANGES_MADE=true
        fi
    else
        echo "$file" > "$GITIGNORE_FILE"
        echo "  Created .gitignore with: $file"
        CHANGES_MADE=true
    fi
    
    # Add reference to README if it exists
    if [ -f "$README_FILE" ]; then
        # Check if section exists
        if ! grep -q "## Large Files Ignored" "$README_FILE"; then
            echo "" >> "$README_FILE"
            echo "## Large Files Ignored" >> "$README_FILE"
            echo "" >> "$README_FILE"
            echo "The following files exceed ${MAX_FILE_SIZE_MB}MB and are excluded from version control:" >> "$README_FILE"
            echo "" >> "$README_FILE"
        fi
        
        # Check if file is already documented
        if ! grep -q "^-\s*$file" "$README_FILE"; then
            FILE_SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            FILE_SIZE_MB=$(awk "BEGIN {printf \"%.2f\", $FILE_SIZE / 1024 / 1024}")
            echo "- \`$file\` (${FILE_SIZE_MB}MB)" >> "$README_FILE"
            echo "  Added to README: $file"
            CHANGES_MADE=true
        fi
    fi
done

# Stage the updated .gitignore and README if they were modified
if [ "$CHANGES_MADE" = true ]; then
    if [ -f "$GITIGNORE_FILE" ]; then
        git add "$GITIGNORE_FILE"
    fi
    if [ -f "$README_FILE" ]; then
        git add "$README_FILE"
    fi
    echo -e "\n${GREEN}Updated .gitignore and README.md have been staged.${NC}"
fi

# Exit with error to prevent commit
echo -e "\n${RED}Commit aborted:${NC} Files exceeding ${MAX_FILE_SIZE_MB}MB were detected and unstaged."
echo "Please review the changes and commit again if needed."
exit 1

