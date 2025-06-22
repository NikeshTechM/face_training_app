import os
import re
import json
import requests
import logging
import math
import PIL.Image
import argparse
from sklearn import neighbors
import pickle
import face_recognition
import numpy as np
from tqdm import tqdm
from io import BytesIO

# -----------------------------
# Utility: Image File Filter
# -----------------------------
def image_files_in_folder(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('png', 'jpg', 'jpeg'))
    ]

# -----------------------------
# Function to Download User Images
# -----------------------------
def fetch_and_download_users(api_url, headers, output_base, dry_run=False):
    logging.info("Fetching data from API...")

    try:
        response = requests.post(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        logging.info("Data fetched successfully.")
        logging.info(f"Raw response: {json.dumps(data, indent=2)}")
    except Exception as e:
        logging.error(f"Failed to fetch API data: {e}")
        return {}

    user_image_data = {}
    os.makedirs(output_base, exist_ok=True)

    if data.get("status") != "success":
        logging.error("API returned non-success status.")
        return {}

    users = data.get("data", {}).get("users", {})

    for user_id, info in users.items():
        logging.debug(f"Inspecting user ID: {user_id}")
        status = info.get("Status", "").lower()
        if status != "pending":
            logging.debug(f"Skipping user {user_id} with status: {status}")
            continue

        name = info.get("Name", "Unknown").replace(" ", "_")
        folder_name = f"{user_id}_{name}"
        folder_path = os.path.join(output_base, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        image_paths = []
        logging.info(f"Processing user {user_id} ({name})")

        for img in info.get("images", []):
            img_url = img["url"]
            original_img_name = img["image"]
            try:
                if dry_run:
                    logging.debug(f"DRY-RUN: Would download {img_url}")
                    continue

                logging.debug(f"Downloading {img_url}")
                img_response = requests.get(img_url)
                img_response.raise_for_status()

                image = PIL.Image.open(BytesIO(img_response.content))
                img_format = image.format.lower()
                base_name = os.path.splitext(original_img_name)[0]
                img_name = f"{base_name}.{img_format}"
                img_path = os.path.join(folder_path, img_name)
                image.save(img_path)
                logging.info(f"Downloaded and saved {img_name}")
                image_paths.append(img_path)

            except Exception as e:
                logging.warning(f"Failed to download or process {img_url}: {e}")

        user_image_data[user_id] = {
            "Name": name,
            "Image": image_paths,
            "Status": status,
        }

    return user_image_data

# -----------------------------
# Function to Train User faces
# -----------------------------
def train_incremental(image_dir,model_save_path, face_data_path="face_data.pkl", n_neighbors=None, knn_algo='ball_tree'):
    X, y = [], []
    image_path = image_dir

    if os.path.exists(face_data_path):
        with open(face_data_path, 'rb') as f:
            previous_data = pickle.load(f)
            X = previous_data['encodings']
            y = previous_data['labels']
        logging.info(f"Loaded {len(X)} existing encodings from {face_data_path}")

    user_dirs = [d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]

    for label in tqdm(user_dirs, desc="Training users"):
        person_dir = os.path.join(image_path, label)
        name_only = re.sub(r'[^a-zA-Z0-9 ]', '', label.split('_', 1)[1])
        user_id = label.split('_', 1)[0]

        for img_path in image_files_in_folder(person_dir):
            try:
                image = PIL.Image.open(img_path).convert('RGB')
                image_np = np.array(image)
                face_locations = face_recognition.face_locations(image_np)
                encodings = face_recognition.face_encodings(image_np, face_locations, model='large')

                if encodings:
                    X.append(encodings[0])
                    y.append(name_only)
                    logging.info(f"Trained {len(encodings)} faces for user: {name_only}")

                else:
                    logging.warning(f"No face found in {img_path}, deleting image")
                    os.remove(img_path)


            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")

    if not X:
        logging.error("No encodings found. Training aborted.")
        return None

    with open(face_data_path, 'wb') as f:
        pickle.dump({'encodings': X, 'labels': y}, f)
    logging.info(f"Saved updated face data to {face_data_path}")

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        logging.info(f"Auto-selected n_neighbors: {n_neighbors}")

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    logging.info(f"KNN training completed. for {y}")

    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)
    logging.info(f"Model saved to {model_save_path}")

    return knn_clf

# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face recognition pipeline")
    parser.add_argument('--dry-run', action='store_true', help='Run without downloading images')
    parser.add_argument('--output-dir', type=str, default='shared', help='Directory for downloads and models')
    parser.add_argument('--only-download', action='store_true', help='Only download images, skip training')
    parser.add_argument('--only-train', action='store_true', help='Only train, skip downloading')
    parser.add_argument('--api-url', type=str, default='https://r6zyaxfjoa.execute-api.us-east-1.amazonaws.com/dev/RHOSimageidentifyfornewdriver', help='API URL for fetching image data')
    parser.add_argument('--headers', type=str, default='{"Content-Type": "application/json"}', help='HTTP headers as JSON string')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--n-neighbors', type=int, default=None, help='Number of neighbors for KNN')
    parser.add_argument('--output-json', type=str, default='user-data.json', help='File to store image metadata')
    parser.add_argument('--knn-algo', type=str, default='ball_tree', help='KNN algorithm to use')
    parser.add_argument('--train-folder-name', type=str, default='train_images', help='Train folder name')

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='[%(levelname)s] %(message)s')

    face_data_path = os.path.join(args.output_dir, 'face_data.pkl')
    model_save_path = os.path.join(args.output_dir, 'trained_knn_model.clf')
    output_json_path = os.path.join(args.output_dir, args.output_json)
    image_dir = os.path.join(args.output_dir, args.train_folder_name)

    if args.only_download and args.only_train:
        logging.error("You cannot use --only-download and --only-train together.")
        exit(1)

    user_data = {}
    if not args.only_train:
        user_data = fetch_and_download_users(
            api_url=args.api_url,
            headers=json.loads(args.headers),
            output_base=image_dir,
            dry_run=args.dry_run
        )

    if not args.only_download:
        train_incremental(
            image_dir= image_dir,
            model_save_path=model_save_path,
            face_data_path=face_data_path,
            n_neighbors=args.n_neighbors,
            knn_algo=args.knn_algo
        )

    # Remove images without faces from user_data JSON
    if user_data:
        for user_id, info in user_data.items():
            valid_images = []
            for img_path in info["Image"]:
                if os.path.exists(img_path):
                    valid_images.append(img_path)
            info["Image"] = valid_images

        with open(output_json_path, 'w') as f:
            json.dump(user_data, f, indent=2)
        logging.info(f"User image metadata saved to {output_json_path}")

    logging.info("Pipeline complete.")
