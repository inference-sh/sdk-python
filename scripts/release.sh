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
VERSION=${LATEST_TAG#v}

# Update version in pyproject.toml
sed -i "s/^version = \".*\"/version = \"${VERSION}\"/" pyproject.toml

# Commit the pyproject.toml change
git add pyproject.toml
git commit -m "chore: sync pyproject.toml version with tag ${LATEST_TAG}"
git push origin main

# Create GitHub release
echo "Creating GitHub release..."
gh release create "${LATEST_TAG}" \
    --title "Release ${LATEST_TAG}" \
    --generate-notes

echo "Released ${LATEST_TAG} successfully!"
