#!/bin/bash
set -e

# Check if version type is provided
if [ -z "$1" ]; then
    echo "Please provide version type: major, minor, or patch"
    exit 1
fi

VERSION_TYPE=$1

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

# Create and push tag
git tag -a "v${NEW_VERSION}" -m "Release v${NEW_VERSION}"
git push origin main --tags

# Create GitHub release
echo "Creating GitHub release..."
gh release create "v${NEW_VERSION}" \
    --title "Release v${NEW_VERSION}" \
    --generate-notes

echo "Released v${NEW_VERSION} successfully!"
