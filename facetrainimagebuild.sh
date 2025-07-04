#!/bin/bash

# === Variables ===
IMAGE_NAME="quay.io/nikesh_sar/face_training_app"
TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

USERNAME="nikesh_sar"
PASSWORD="Nikesh@123"

# Build directory where the Dockerfile exists
# BUILD_DIR="/root/ocpcls/newconfig0405/oclogs/face_training_app"
# For Autosd
 BUILD_DIR="/root/ocppipeline/oclogs/face_training_app"

# API endpoint to notify
API_URL="https://sosly6i1zl.execute-api.us-east-1.amazonaws.com/dev/RHOSGetFluentBitLogs"

# === Ensure Log Directory Exists ===
mkdir -p /var/log/podman
chmod 755 /var/log/podman

BUILD_LOG="/var/log/podman/build.log"

# === Delete existing build log file and create a new empty one ===
if [ -f "$BUILD_LOG" ]; then
  rm "$BUILD_LOG"
fi
touch "$BUILD_LOG"

# === Dockerfile Existence Check ===
if [ ! -f "$BUILD_DIR/Dockerfile" ]; then
  echo "❌ Dockerfile not found in $BUILD_DIR" | tee -a "$BUILD_LOG"
  exit 1
fi

# === Login to Quay.io (optional if push not needed) ===
echo "$PASSWORD" | podman login quay.io -u "$USERNAME" --password-stdin >> "$BUILD_LOG" 2>&1

# === Remove all existing local images with IMAGE_NAME (any tag) ===
#EXISTING_IMAGE_IDS=$(podman images --format "{{.ID}}" "$IMAGE_NAME")

#if [ -n "$EXISTING_IMAGE_IDS" ]; then
#  echo "🔍 Found existing images for $IMAGE_NAME. Removing..." | tee -a "$BUILD_LOG"
#  podman rmi -f $EXISTING_IMAGE_IDS >> "$BUILD_LOG" 2>&1
#  echo "✅ Existing images removed." | tee -a "$BUILD_LOG"
#else
#  echo "ℹ️ No existing images found for $IMAGE_NAME. Proceeding to build..." | tee -a "$BUILD_LOG"
#fi

# === Build Image with tag 'latest' ===
{
  echo "=== Build Started at $(date -Iseconds) ==="
  podman build -t "$FULL_IMAGE" "$BUILD_DIR"
  BUILD_STATUS=$?
  echo "=== Build Finished at $(date -Iseconds) ==="
} >> "$BUILD_LOG" 2>&1

# Check build success
if [ $BUILD_STATUS -ne 0 ]; then
  echo "❌ Build failed. Check $BUILD_LOG for details." | tee -a "$BUILD_LOG"
  exit 1
fi

echo "Successfully built new image: $FULL_IMAGE" | tee -a "$BUILD_LOG"

#Test and validate the container
python3 /root/ocpcls/validation-v2/master.py --op detection

#Push the container to Quay Repo
# /root/ocpcls/newconfig0405/oclogs/pushimagetoquay-facetraining.sh
# For Autosd
/root/ocppipeline/oclogs/pushimagetoquay-facetraining.sh
