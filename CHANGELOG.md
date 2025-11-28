# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-11-28

### Added

- **Configuration Persistence**: Settings (database path, folder path) are now saved to `siglip2_config.json` and persist between sessions.
- **Image Preview Controls**: Added Pan (drag) and Zoom (mouse wheel) functionality to the image preview canvas.
- **Rescoring**: New "Rescore" feature allows refining search results with a secondary text query.
- **Batch Processing**: Image indexing is now batched (default size: 10) for improved performance.
- **GUI Improvements**:
  - Dedicated "Load Model" button.
  - Clear separation between "Index" and "Search" workflows.
  - Improved status feedback.
- **Recursive Indexing**: Now uses `rglob` to find images and videos in all subdirectories.

### Changed

- **Script Name**: Renamed main script from `sigLIP2_en.py` to `S2N_Search.py`.
- **Database Schema**: Updated schema for better compatibility.
- **Video Indexing**: Simplified video frame extraction to 'uniform' method for consistency.

### Removed

- **Legacy Video Methods**: Removed 'start', 'mid', 'end' extraction methods in favor of 'uniform'.
- **Threshold Slider**: Removed UI slider; thresholding is now handled internally or via code if needed.
