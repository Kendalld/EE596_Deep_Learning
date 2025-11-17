# Pre-commit Hook for Large Files

This directory contains a generic pre-commit hook system that prevents committing files larger than a specified size limit (default: 50MB).

## Files

- `pre-commit-large-files.sh` - The actual pre-commit hook script
- `install-pre-commit-hook.sh` - Installation script to set up the hook in your repository

## Quick Start

1. Install the hook:
   ```bash
   ./scripts/install-pre-commit-hook.sh
   ```

2. Try committing a large file - the hook will automatically:
   - Unstage the large file
   - Add it to `.gitignore`
   - Document it in `README.md`
   - Abort the commit

## How It Works

When you attempt to commit files:

1. The hook checks all staged files for size
2. Files exceeding the limit are automatically:
   - Unstaged from the commit
   - Added to `.gitignore` (if not already present)
   - Documented in `README.md` under "Large Files Ignored" section
3. The updated `.gitignore` and `README.md` are staged
4. The commit is aborted so you can review the changes

## Configuration

The hook can be configured via environment variables:

- `MAX_FILE_SIZE_MB` - Maximum file size in MB (default: 50)
- `README_FILE` - Path to README file (default: README.md)
- `GITIGNORE_FILE` - Path to .gitignore file (default: .gitignore)

### Setting Custom Limits

**During installation:**
```bash
MAX_FILE_SIZE_MB=100 ./scripts/install-pre-commit-hook.sh
```

**After installation:**
Edit `.git/hooks/pre-commit` and modify the `MAX_FILE_SIZE_MB` variable at the top.

**Per-commit override:**
```bash
MAX_FILE_SIZE_MB=100 git commit -m "message"
```

## Using in Other Repositories

This hook is designed to be generic and portable:

1. Copy both scripts to your repository:
   ```bash
   cp scripts/pre-commit-large-files.sh /path/to/other/repo/scripts/
   cp scripts/install-pre-commit-hook.sh /path/to/other/repo/scripts/
   ```

2. Run the installation script from the new repository:
   ```bash
   cd /path/to/other/repo
   ./scripts/install-pre-commit-hook.sh
   ```

The hook is self-contained and only requires standard Unix tools (bash, git, stat, awk, grep).

## Troubleshooting

**Hook not running:**
- Ensure the hook is executable: `chmod +x .git/hooks/pre-commit`
- Verify it's installed: `ls -la .git/hooks/pre-commit`

**False positives:**
- Adjust `MAX_FILE_SIZE_MB` to a higher value
- Or temporarily bypass: `git commit --no-verify`

**README not updating:**
- Ensure `README.md` exists in the repository root
- Check file permissions

