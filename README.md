# Environment Setup

```bash
conda create -n api_sed python=3.9
conda activate api_sed
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_sed.txt
```

---

# Usage

## Start the Server

```bash
uvicorn api_endpoint:app --host 0.0.0.0 --port 8000
```

## Send a Request

```bash
curl -X POST "http://localhost:8000/detect?threshold=0.5" \
-F "files=@/path/to/local/file1.mp3" \
-F "files=@/path/to/local/file2.mp3"
```

---

# Notes

1. Please replace the file paths with the actual local paths on your system (please ensure not to remove the @, the filepath should follow the @ symbol).

2. The threshold value is configurable in the query string (?threshold=0.5). Please modify this number to adjust the detection sensitivity.

curl -X POST "http://localhost:8000/detect?threshold=0.5" \
-F "files=@/home/users/ntu/bhargavi/crowd.mp3" \
-F "files=@/home/users/ntu/bhargavi/manshout.mp3"