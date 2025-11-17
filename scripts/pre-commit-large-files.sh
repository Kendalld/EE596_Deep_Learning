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
NEEDS_ATTENTION=false
UNHANDLED_FILES=()
LARGE_FILES=()
LARGE_FILE_STATUSES=()
STAGED_PATHS=()
STAGED_STATUSES=()

# Read staged files plus their status (A/M/C)
while IFS=$'\t' read -r status file; do
    if [ -z "$file" ]; then
        continue
    fi
    STAGED_STATUSES+=("$status")
    STAGED_PATHS+=("$file")
done < <(git diff --cached --name-status --diff-filter=ACM)

if [ ${#STAGED_PATHS[@]} -eq 0 ]; then
    exit 0
fi

echo "Checking staged files for size limit (${MAX_FILE_SIZE_MB}MB)..."

# Check each staged file
for idx in "${!STAGED_PATHS[@]}"; do
    file="${STAGED_PATHS[$idx]}"
    status="${STAGED_STATUSES[$idx]}"

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
        LARGE_FILE_STATUSES+=("$status")
    fi
done

# If no large files found, exit successfully
if [ ${#LARGE_FILES[@]} -eq 0 ]; then
    exit 0
fi

# Process large files
echo -e "\n${YELLOW}Processing large files...${NC}"

for idx in "${!LARGE_FILES[@]}"; do
    file="${LARGE_FILES[$idx]}"
    status="${LARGE_FILE_STATUSES[$idx]}"

    case "$status" in
        A|C)
            # Added/copied files can be safely ignored
            ;;
        *)
            echo "  Cannot auto-ignore tracked file: $file"
            UNHANDLED_FILES+=("$file")
            NEEDS_ATTENTION=true
            continue
            ;;
    esac

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

if [ "$NEEDS_ATTENTION" = true ]; then
    echo -e "\n${RED}Commit blocked:${NC} The following tracked files exceed ${MAX_FILE_SIZE_MB}MB:"
    for file in "${UNHANDLED_FILES[@]}"; do
        echo "  - $file"
    done
    echo "Remove them from history or Git LFS before committing."
    exit 1
fi

echo -e "\n${GREEN}Large files were automatically ignored. Commit will continue without them.${NC}"
exit 0
