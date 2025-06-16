
# Face Recognition App

This project downloads user images from an API, processes faces using `face_recognition`, and trains a KNN model for face recognition. It supports running in a container using Podman with flexible options for downloading, training, or both.

---

## Project Structure

```

face-recognition-app/
├── app.py               # Main application script
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container build instructions (works with Podman too)
├── shared\_data/         # Data directory for images, models, metadata (created at runtime)
└── README.md            # This documentation file

````

---

## Build Container Image (Podman)

```bash
podman build -t face-recog-app .
````

---

## Run Container (Podman)

Mount a local directory (`shared_data`) to persist data outside the container:

```bash
podman run --rm -v $(pwd)/shared_data:/app/shared_data face-recog-app [options]
```

---

## Usage Examples

* **Download images only (no training):**

```bash
podman run --rm -v $(pwd)/shared_data:/app/shared_data face-recog-app --only-download --output-dir shared_data
```

* **Train model only (skip downloading):**

```bash
podman run --rm -v $(pwd)/shared_data:/app/shared_data face-recog-app --only-train --output-dir shared_data
```

* **Download images and train model:**

```bash
podman run --rm -v $(pwd)/shared_data:/app/shared_data face-recog-app --download-and-train --output-dir shared_data
```

* **Dry run mode (simulate downloads):**

```bash
podman run --rm -v $(pwd)/shared_data:/app/shared_data face-recog-app --only-download --dry-run --output-dir shared_data
```

---

## Command Line Arguments / Parsers

| Option                 | Description                                                                     | Default                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `--only-download`      | Download images only; skip training                                             | False                                                                                      |
| `--only-train`         | Train model only; skip downloading                                              | False                                                                                      |
| `--download-and-train` | Perform both downloading and training                                           | False                                                                                      |
| `--output-dir`         | Directory inside container to save downloaded images, model files, and metadata | `shared_data`                                                                              |
| `--dry-run`            | Run without actually downloading images                                         | False                                                                                      |
| `--api-url`            | API endpoint to fetch user image metadata                                       | `https://r6zyaxfjoa.execute-api.us-east-1.amazonaws.com/dev/RHOSimageidentifyfornewdriver` |
| `--headers`            | HTTP headers for API requests, as JSON string                                   | `{"Content-Type": "application/json"}`                                                     |
| `--log-level`          | Logging verbosity level (DEBUG, INFO, WARNING, ERROR)                           | INFO                                                                                       |
| `--n-neighbors`        | Number of neighbors for KNN classifier                                          | Auto-calculated (sqrt of data size)                                                        |
| `--output-json`        | Filename to save user image metadata                                            | `user_images.json`                                                                         |
| `--knn-algo`           | Algorithm for KNN classifier (`ball_tree`, `kd_tree`, etc.)                     | `ball_tree`                                                                                |
| `--limit-users`        | Limit the number of users to process from the API (useful for testing)          | None                                                                                       |

---

## Data & Output Explanation

* **`shared_data/`**: Root directory for all outputs, mounted from host.
* **`shared_data/downloaded_users/`**: Subfolder containing downloaded images, organized by `<userID>_<Name>`.
* **`shared_data/face_data.pkl`**: Pickle file storing face encodings for incremental training.
* **`shared_data/trained_knn_model.clf`**: Serialized KNN model saved after training.
* **`shared_data/user_images.json`**: Metadata JSON describing downloaded images.

---

## Local Development (Without Container)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run app locally with options, for example:

```bash
python app.py --only-download --output-dir shared_data
```

---


## Author

*AYUSH*




