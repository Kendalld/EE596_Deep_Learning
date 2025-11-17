cd#!/bin/bash
#
# Installation script for the pre-commit hook that prevents large files.
# This script copies the hook to .git/hooks/pre-commit and makes it executable.
#
# Usage:
#   ./scripts/install-pre-commit-hook.sh
#   MAX_FILE_SIZE_MB=100 ./scripts/install-pre-commit-hook.sh  # Custom size limit
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOK_SOURCE="$SCRIPT_DIR/pre-commit-large-files.sh"
HOOK_TARGET="$REPO_ROOT/.git/hooks/pre-commit"

# Check if we're in a git repository
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "Error: Not in a git repository. Please run this from the repository root."
    exit 1
fi

# Check if hook source exists
if [ ! -f "$HOOK_SOURCE" ]; then
    echo "Error: Hook script not found at $HOOK_SOURCE"
    exit 1
fi

# Make hook source executable
chmod +x "$HOOK_SOURCE"

# Copy hook to git hooks directory
cp "$HOOK_SOURCE" "$HOOK_TARGET"

# If MAX_FILE_SIZE_MB is set, add it to the hook
if [ -n "$MAX_FILE_SIZE_MB" ]; then
    # Replace the default value in the hook
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/MAX_FILE_SIZE_MB=\${MAX_FILE_SIZE_MB:-50}/MAX_FILE_SIZE_MB=\${MAX_FILE_SIZE_MB:-$MAX_FILE_SIZE_MB}/" "$HOOK_TARGET"
    else
        sed -i "s/MAX_FILE_SIZE_MB=\${MAX_FILE_SIZE_MB:-50}/MAX_FILE_SIZE_MB=\${MAX_FILE_SIZE_MB:-$MAX_FILE_SIZE_MB}/" "$HOOK_TARGET"
    fi
    echo "Installed pre-commit hook with MAX_FILE_SIZE_MB=$MAX_FILE_SIZE_MB"
else
    echo "Installed pre-commit hook with default MAX_FILE_SIZE_MB=50"
fi

chmod +x "$HOOK_TARGET"

echo "Pre-commit hook installed successfully at $HOOK_TARGET"
echo ""
echo "To customize the size limit, edit $HOOK_TARGET and set MAX_FILE_SIZE_MB"
echo "Or set it when installing: MAX_FILE_SIZE_MB=100 ./scripts/install-pre-commit-hook.sh"

