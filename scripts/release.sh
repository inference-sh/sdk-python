#!/bin/bash
set -e

# Check if version type is provided
if [ -z "$1" ]; then
    echo "Please provide version type: major, minor, or patch"
    exit 1
fi

VERSION_TYPE=$1

# Store the original version before any changes
ORIGINAL_VERSION=$(poetry version -s)

# Function to rollback changes
rollback() {
    echo "Error occurred. Rolling back changes..."
    # Reset version in pyproject.toml
    poetry version "$ORIGINAL_VERSION"
    # Remove local tag if it exists
    if git tag | grep -q "v${NEW_VERSION}"; then
        git tag -d "v${NEW_VERSION}"
    fi
    # Reset any changes in git
    git reset --hard HEAD^
    echo "Rollback complete. Version restored to ${ORIGINAL_VERSION}"
    exit 1
}

# Set up trap to catch errors
trap rollback ERR

# Ensure we're on main branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo "Please switch to main branch first"
    exit 1
fi

# Ensure working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "Working directory is not clean. Please commit or stash changes first."
    exit 1
fi

# Pull latest changes
git pull origin main

# Bump version using poetry
echo "Bumping $VERSION_TYPE version..."
poetry version $VERSION_TYPE
NEW_VERSION=$(poetry version -s)

# Update files
git add pyproject.toml

# Commit version bump
git commit -m "chore: bump version to ${NEW_VERSION}"

# Push commit
git push origin main

# Create and push only the new tag
git tag "v${NEW_VERSION}"
git push origin "v${NEW_VERSION}"

# Create GitHub release
echo "Creating GitHub release..."
gh release create "v${NEW_VERSION}" \
    --title "Release v${NEW_VERSION}" \
    --generate-notes

echo "Released v${NEW_VERSION} successfully!"

# Remove the trap since we succeeded
trap - ERR
