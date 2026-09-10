# Robo Dog — Face Detection & Recognition

A background service that pulls a live video stream from an ESP32-CAM, detects faces with YOLO, and identifies each face in-memory against a known-faces database — saving only genuinely new unknown faces to disk.

## How It Works

1. **`camera.py`** connects to the ESP32-CAM stream, reads frames continuously, and runs YOLO face detection on each one.
2. Every detected face crop is passed **directly in memory** to `facerecog.identify_face()` — nothing is written to disk at the detection stage.
3. **`facerecog.py`** owns the full decision for each face:
   - Compares it against the **known-faces database** (photos you enroll ahead of time). A match discards the face — nothing is saved.
   - If it's not a known person, compares it against faces already saved in the **unknown-faces folder** (using embedding similarity, not a perceptual hash).
   - A similar-enough unknown face is treated as a **duplicate** and skipped.
   - A genuinely new unknown face is **saved** to the unknown-faces folder and its embedding is remembered for future dedup.
4. **`main.py`** is the coordinator — it initializes face recognition, starts the camera loop on a background thread, and prints periodic status updates (faces detected, known matches, new/duplicate unknowns).

Face recognition is no longer a separate polling loop — it's initialized once at startup and called synchronously inside the camera loop for every detected face.

## Requirements

- Python 3.9+
- An ESP32-CAM (or any MJPEG/HTTP video stream) reachable on your network
- A folder of enrolled photos, one face per image, for the known-faces database

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- `ultralytics` — YOLO face detection (also pulls in torch/torchvision)
- `opencv-python` — video capture and image I/O
- `numpy` — embedding math
- `insightface` + `onnxruntime` — face embeddings and matching
- `python-dotenv` — loads configuration from `.env`

You'll also need a YOLO face-detection weights file named **`blacknwhite.pt`** in the project root (referenced by `camera.py`).

## Configuration

Create a `.env` file in the project root with:

```env
videourl=http://<esp32-cam-ip>/stream
kface=/path/to/known_faces
uface=/path/to/unknown_faces
```

| Variable   | Purpose                                                              |
|------------|-----------------------------------------------------------------------|
| `videourl` | URL of the camera stream (ESP32-CAM snapshot/stream endpoint)         |
| `kface`    | Folder of known-face photos, one clear face per file (filename → name)|
| `uface`    | Folder where new unknown faces get saved automatically                |

**Known faces:** name each image after the person, e.g. `john_doe.jpg` → enrolled as "John Doe" (underscores/hyphens become spaces, title-cased).

**Unknown faces:** this folder is managed automatically — files are saved as `<n>_unknown.jpg` and reloaded on restart so deduplication keeps working across runs.

## Usage

```bash
python main.py
```

You'll get an interactive command prompt:

| Command  | Action                                                        |
|----------|-----------------------------------------------------------------|
| `start`  | Load face recognition (if needed) and start the camera thread  |
| `stop`   | Stop the camera thread                                          |
| `status` | Print current stats (faces detected, matches, unknowns, etc.)  |
| `help`   | Show the command list                                           |
| `exit`   | Stop everything and quit                                        |

Every 10 seconds while running, the coordinator prints a status block showing total faces detected, known matches, new unknown faces saved, and duplicate unknowns skipped.

## Key Thresholds (in `facerecog.py`)

| Constant                    | Default | Meaning                                                        |
|------------------------------|---------|------------------------------------------------------------------|
| `MATCH_THRESHOLD`             | 0.30    | Cosine similarity to count as a match against a known person    |
| `UNKNOWN_DEDUP_THRESHOLD`     | 0.35    | Similarity to treat an unknown face as "the same stranger" (stricter, since a false skip just costs one photo) |
| `MIN_FACE_SIZE`               | 120 px  | Faces smaller than this are upscaled before embedding            |

And in `camera.py`:

| Constant          | Default | Meaning                                  |
|--------------------|---------|--------------------------------------------|
| `CONF_THRESHOLD`   | 0.5     | YOLO detection confidence threshold        |
| `SNAPSHOT_DELAY`   | 0.1 s   | Delay between processed frames             |

## Project Structure

```
.
├── main.py            # Coordinator: starts/stops the pipeline, prints status
├── camera.py           # ESP32-CAM capture + YOLO face detection
├── facerecog.py         # InsightFace embeddings, known-face matching, unknown-face dedup
├── requirements.txt     # Python dependencies
├── blacknwhite.pt       # YOLO face-detection weights (not included — supply your own)
└── .env                 # Local configuration (videourl, kface, uface) — not committed
```

## Notes

- All shared counters in `camera.py` are protected with locks, so status reads are safe while the capture thread is running.
- If the camera stream drops, `camera.py` automatically retries the connection with a short backoff.
- If the camera thread dies unexpectedly, the coordinator's status monitor detects it and stops itself.
