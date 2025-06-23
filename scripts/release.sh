#!/bin/bash
set -e

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

# Get the latest tag
LATEST_TAG=$(git describe --tags --abbrev=0)

# Create GitHub release
echo "Creating GitHub release for ${LATEST_TAG}..."
gh release create "${LATEST_TAG}" \
    --title "Release ${LATEST_TAG}" \
    --generate-notes

echo "Released ${LATEST_TAG} successfully!"
