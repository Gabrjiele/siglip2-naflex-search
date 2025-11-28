# SigLIP 2 NaFlex Image and Video Search Tool

A natural language image and video search tool powered by [Google's SigLIP 2 model](https://huggingface.co/google/siglip2-so400m-patch16-naflex) with NaFlex architecture. Search through your local image and video collections using natural language queries or image similarity.
<br>
<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/89008d71-d2d8-4091-b409-44bf6fdb24f4" width="90%">
</p>
<br>
<br>

## Features

- **Natural language search**: Find images and videos using text descriptions
- **Image-to-Image search**: Search using reference images
- **Video support**: Extract and index video frames for content search
- **Dual interface**: Available both as GUI and CLI
- **GPU acceleration**: Supports CUDA, DirectML (AMD/Intel), and CPU
- **Flexible patch sizes**: From 128 to 1024 patches for resolution control
- **SQLite database**: Efficient embedding storage with manual cleanup

## Installation

### Setup

1. Either clone or download the repository in any directory (it doesn’t have to contain pictures):
```bash
git clone https://github.com/Gabrjiele/siglip2-naflex-search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Needed for DirectML support on AMD/Intel GPUs:
```bash
pip install torch-directml
```

## Usage

### GUI Mode

Simply run the script without arguments:

```bash
python S2N_Search.py
```

The GUI provides:
- Database and folder selection
- Model configuration (patches, video settings)
- Interactive search with preview
- Result visualization with similarity scores

Example queries:
<table align="center" width="100%">
  <tr>
    <td width="20%"><img src="https://github.com/user-attachments/assets/dc58324a-a315-471f-9791-37aa981c7939" style="width: 100%; height: auto;" alt="Immagine 1"></td>
    <td width="20%"><img src="https://github.com/user-attachments/assets/0ec7833e-cb60-4e43-a149-049ffd7f72ce" style="width: 100%; height: auto;" alt="Immagine 2"></td>
    <td width="20%"><img src="https://github.com/user-attachments/assets/d1fe1441-7d04-4718-b5c0-4bf83c0d0f41" style="width: 100%; height: auto;" alt="Immagine 3"></td>
    <td width="20%"><img src="https://github.com/user-attachments/assets/d15260f6-b74c-4136-b74c-c3dd0da2a283" style="width: 100%; height: auto;" alt="Immagine 3"></td>
    <td width="20%"><img src="https://github.com/user-attachments/assets/92e195ea-b274-4157-8791-fc01458ce924" style="width: 100%; height: auto;" alt="Immagine 3"></td>
  </tr>
</table>

<details>
<summary><h4>GUI quick start guide</h4></summary>
  
> **✅ Advice for a more convenient use ✅**  
>  
> Model settings (number of patches, video frame parameters, and images folder) only take effect **during the indexing phase**.  
>  
> Once files are indexed, the embeddings stored in the database remain fixed and will not change if you adjust the settings later.  
>  
> So, for convenience, you can use the search tool without changing these parameters, and only tweak them when indexing new images or re-indexing an existing collection.

**First Time Setup (Creating a new database):**

1. Click **"Select Database/Folder"** → Choose "No" → Select the folder where you want to create the database
2. Click **"Browse"** and select the directory with the images you want to index
3. Click **"1. Load Model"** and wait for it to complete (~20 seconds or less)
4. Click **"2. Index"** to process all files in the selected folder (this may take several time depending on dataset size, files size, selected resolution and hardware)
5. Enter a search query in the "Text" field or click **"Load Query Image..."**
6. Click **"3. Search"** to find matching images/videos

**Searching an existing database:**

1. Click **"Select Database/Folder"** → Choose "Yes" → Select your `.db` file
2. Click **"1. Load Model"**
3. Enter your search query (text, image, or both)
4. Click **"3. Search"**

**Adding New Images to Existing Database:**

1. Load your existing database
2. Click **"1. Load Model"**
3. Browse to select the folder (can be the same or a different one)
4. Click **"2. Index"** → only new/modified files will be processed

#### GUI Features

- **Model Patches**: Higher values (512-1024) give better quality but are slower during indexing of images.
- **Video Settings**: Control how many frames are extracted and from where in the video
- **Threshold Slider**: Filter out low-similarity results (0.0 = show all, higher = more selective)
- **Cleanup Database**: Remove entries for deleted files to keep the database clean
- **Result Indicators**: `*` = excellent match (≥0.8), `-` = good match (≥0.6), `.` = lower match

</details>

<details>
<summary><h3>CLI Mode</h3></summary>

Index a folder:
```bash
python S2N_Search.py --index /path/to/images --db my_database.db
```

Search by text:
```bash
python S2N_Search.py --search-text "sunset over mountains" --db my_database.db --top-k 20
```

Search by image:
```bash
python S2N_Search.py --search-image /path/to/query.jpg --db my_database.db
```

Combined search (multimodal):
```bash
python S2N_Search.py --search-text "red car" --search-image car.jpg --db my_database.db
```

#### CLI Options

- `--index PATH`: Folder to index
- `--search-text TEXT`: Text query
- `--search-image PATH`: Image query path
- `--top-k N`: Number of results (default: 10)
- `--db PATH`: Database file path (default: siglip2_embeddings.db)
- `--device {cuda,cpu,dml}`: Force specific device
- `--max-patches N`: Max patches for model (default: 256)
- `--cleanup`: Remove orphaned embeddings

</details>

## Video Support

> [!NOTE]
> **Video Analysis Beta**: Video search is currently experimental. It works by averaging embeddings from a few frames, which means it captures the general "vibe" or dominant content but lacks granular temporal understanding (e.g., finding a specific action at a specific second). Improved temporal analysis is planned for future updates.

The tool supports common video formats:
- MP4, AVI, MOV, MKV, FLV, WMV, WebM

Videos are indexed by extracting representative frames using the `uniform` method (evenly distributed frames across video).

<details>
<summary><h2>Model configuration</h2></summary>

The tool uses Google's `siglip2-so400m-patch16-naflex` model with configurable patch counts:

- **128-256 patches**: Fast, good for general use
- **512 patches**: Balanced speed/quality
- **1024 patches**: Maximum quality for high-resolution images

</details>

<details>
<summary><h2>Database management</h2></summary>

Embeddings are stored in SQLite with automatic management:
- Only modified or new files are re-indexed
- Deleted files can be cleaned up with `--cleanup` flag
- Each entry stores filepath, timestamp, embedding, and file type

</details>

<details>
<summary><h2>Performance tips</h2></summary>

- Start with 256 patches and increase if needed
- Use cleanup regularly to remove orphaned entries
- Index incrementally → only new/modified files are processed

</details>

<details>
<summary><h2>Technical details</h2></summary>

- **Model**: Google SigLIP 2 with NaFlex architecture
- **Embedding size**: 1152
- **Similarity metric**: Cosine similarity (dot product of normalized vectors)
- **Database**: SQLite with BLOB storage for embeddings
- **Image processing**: PIL with decompression bomb protection disabled
- **Video processing**: OpenCV (cv2) for frame extraction

</details>

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Google for the SigLIP 2 model

## Contributing

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

## Author

Created by Peris Gabriele
