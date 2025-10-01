#!/usr/bin/env python3

"""
SigLIP 2 NaFlex image and video search tool
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import numpy as np
import torch
from transformers import AutoProcessor, Siglip2Model
import sqlite3
import warnings
import threading
from tqdm import tqdm
import cv2

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

try:
    import torch_directml
except ImportError:
    torch_directml = None

# Core functions

def setup_device(forced_device=None):
    """Device detection"""
    if forced_device:
        if forced_device == "dml" and (torch_directml is None or not torch_directml.is_available()):
            return "cpu", "DirectML not available. Falling back to CPU."
        if forced_device == "cuda" and not torch.cuda.is_available():
            return "cpu", "CUDA not available. Falling back to CPU."
        return forced_device, f"Device forced to {forced_device}."
    if torch.cuda.is_available():
        return "cuda", "NVIDIA GPU (CUDA) detected."
    if torch_directml and torch_directml.is_available():
        return "dml", "AMD/Intel GPU (DirectML) detected."
    return "cpu", "No compatible GPU found. Using CPU."

def init_db(db_path):
    """Initialize the database"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            filepath TEXT PRIMARY KEY,
            modified_at REAL NOT NULL,
            embedding BLOB NOT NULL,
            model_version TEXT DEFAULT 'siglip2-naflex',
            file_type TEXT DEFAULT 'image'
        );
        """)

def clear_db(db_path):
    """Clear the database"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        print("Database completely cleared.")

def cleanup_orphaned_entries(db_path, progress_callback=None):
    """
    Cleanup orphaned entries from the database.
    Args:
        db_path: Path to the SQLite database.
        progress_callback: Function to report progress.
    Returns:
        The number of removed entries.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filepath FROM embeddings")
        all_paths = cursor.fetchall()
        
        if progress_callback:
            progress_callback(f"Checking {len(all_paths)} files in the database...")
        
        orphaned_paths = []
        for (path,) in tqdm(all_paths, desc="Verifying file existence"):
            if not os.path.exists(path):
                orphaned_paths.append(path)
        
        if orphaned_paths:
            if progress_callback:
                progress_callback(f"Removing {len(orphaned_paths)} orphaned embeddings...")
            for path in orphaned_paths:
                cursor.execute("DELETE FROM embeddings WHERE filepath=?", (path,))
            conn.commit()
            if progress_callback:
                progress_callback(f"Cleanup complete: removed {len(orphaned_paths)} orphaned embeddings.")
        else:
            if progress_callback:
                progress_callback("No orphaned embeddings found.")
        
        return len(orphaned_paths)

def extract_video_frames(video_path, num_frames=5, method='uniform'):
    """
    Extract frames from a video file.
    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract.
        method: Extraction method ('uniform', 'start', 'mid', 'end').
    Returns:
        A list of frames as RGB images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    frames = []
    
    if method == 'uniform':
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif method == 'start':
        frame_indices = np.arange(min(num_frames, total_frames))
    elif method == 'mid':
        mid = total_frames // 2
        half = num_frames // 2
        frame_indices = np.arange(max(0, mid - half), min(total_frames, mid + half))
    elif method == 'end':
        frame_indices = np.arange(max(0, total_frames - num_frames), total_frames)
    else: # Default to uniform
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames

def normalize_embedding(features):
    """L2 normalize embedding features."""
    return features / features.norm(p=2, dim=-1, keepdim=True)

def index_images_and_videos(folder_path, device, processor, model, db_path, batch_size=8,
                            progress_callback=None, max_num_patches=256,
                            video_frames=5, video_method='uniform'):
    """
    Index all images and videos in a folder and its subdirectories.
    """
    if progress_callback:
        progress_callback("Cleaning database of deleted files...")
    orphaned_count = cleanup_orphaned_entries(db_path, progress_callback)
    if orphaned_count > 0 and progress_callback:
        progress_callback(f"Removed {orphaned_count} orphaned embeddings from the database.")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    abs_folder_path = os.path.abspath(folder_path)
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
    
    all_files = []
    for root, _, files in os.walk(abs_folder_path):
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith(image_extensions):
                all_files.append((os.path.join(root, file), 'image'))
            elif file_lower.endswith(video_extensions):
                all_files.append((os.path.join(root, file), 'video'))
    
    paths_to_process = []
    if progress_callback:
        progress_callback(f"Checking {len(all_files)} files (images and videos)...")
    
    for path, file_type in tqdm(all_files, desc="Checking files"):
        try:
            last_modified = os.path.getmtime(path)
            cursor.execute("SELECT modified_at FROM embeddings WHERE filepath=?", (path,))
            result = cursor.fetchone()
            if not result or result[0] < last_modified:
                paths_to_process.append((path, file_type))
        except FileNotFoundError:
            continue
    
    if not paths_to_process:
        if progress_callback:
            progress_callback("Database is already up to date. No new files to index.")
        conn.close()
        return
    
    if progress_callback:
        progress_callback(f"Indexing {len(paths_to_process)} files with SigLIP 2 NaFlex...")
    
    commit_interval = 50
    
    for i, (path, file_type) in enumerate(tqdm(paths_to_process, desc="Indexing with SigLIP 2")):
        try:
            if file_type == 'image':
                image = Image.open(path).convert("RGB")
                inputs = processor(
                    images=image,
                    return_tensors="pt",
                    max_num_patches=max_num_patches
                ).to(device)
                
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    image_features = normalize_embedding(image_features)
                
                embedding = image_features[0].cpu().numpy().astype(np.float32).tobytes()
                
            elif file_type == 'video':
                frames = extract_video_frames(path, num_frames=video_frames, method=video_method)
                if not frames:
                    print(f"No frames extracted from {path}")
                    continue
                
                frame_embeddings = []
                for frame in frames:
                    frame_pil = Image.fromarray(frame)
                    inputs = processor(
                        images=frame_pil,
                        return_tensors="pt",
                        max_num_patches=max_num_patches
                    ).to(device)
                    
                    with torch.no_grad():
                        frame_features = model.get_image_features(**inputs)
                        frame_features = normalize_embedding(frame_features)
                        frame_embeddings.append(frame_features.cpu().numpy())
                
                avg_embedding = np.mean(frame_embeddings, axis=0).astype(np.float32)
                norm = np.linalg.norm(avg_embedding)
                if norm > 0:
                    avg_embedding = avg_embedding / norm
                embedding = avg_embedding.tobytes()
            
            last_modified = os.path.getmtime(path)
            cursor.execute(
                "REPLACE INTO embeddings (filepath, modified_at, embedding, model_version, file_type) VALUES (?, ?, ?, ?, ?)",
                (path, last_modified, embedding, 'siglip2-naflex', file_type)
            )
            
            if (i + 1) % commit_interval == 0:
                conn.commit()
            
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(f"Indexing: {i+1}/{len(paths_to_process)}")
        
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    conn.commit()
    conn.close()
    
    if progress_callback:
        progress_callback(f"Indexing complete. Processed {len(paths_to_process)} files.")

def get_query_embedding(query_text, query_image_path, device, processor, model, max_num_patches=256):
    """Calculate the embedding for a text or image query."""
    text_embedding, image_embedding = None, None
    
    with torch.no_grad():
        if query_text:
            processed_text = query_text.lower()
            
            inputs = processor(
                text=[processed_text],
                return_tensors="pt",
                padding="max_length",
                max_length=64
            ).to(device)
            text_features = model.get_text_features(**inputs)
            text_features = normalize_embedding(text_features)
            text_embedding = text_features.cpu().numpy().astype(np.float32)
        
        if query_image_path:
            try:
                image = Image.open(query_image_path).convert("RGB")
                inputs = processor(
                    images=image,
                    return_tensors="pt",
                    max_num_patches=max_num_patches
                ).to(device)
                image_features = model.get_image_features(**inputs)
                image_features = normalize_embedding(image_features)
                image_embedding = image_features.cpu().numpy().astype(np.float32)
            except Exception as e:
                print(f"Error processing query image: {e}")
                return None
        
        if text_embedding is not None and image_embedding is not None:
            # Simple averaging for multimodal query
            text_weight = 0.5
            image_weight = 0.5
            combined_embedding = (text_weight * text_embedding + image_weight * image_embedding)
            norm = np.linalg.norm(combined_embedding, axis=1, keepdims=True)
            if norm > 0:
                combined_embedding = combined_embedding / norm
            return combined_embedding
        
        return text_embedding if text_embedding is not None else image_embedding

def search_db(query_embedding, db_path, top_k=10, similarity_threshold=0.0):
    """Search the database for similar embeddings."""
    if query_embedding is None:
        return []
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filepath, embedding, file_type FROM embeddings")
        all_embeddings = cursor.fetchall()
        
        if not all_embeddings:
            return []
        
        filepaths = [row[0] for row in all_embeddings]
        file_types = [row[2] for row in all_embeddings]
        db_embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in all_embeddings])
        
        # Cosine similarity
        similarities = np.dot(db_embeddings, query_embedding.T).squeeze()
        
        if similarities.ndim == 0: # Handle only one item in db
            similarities = np.array([similarities])
        
        valid_indices = np.where(similarities >= similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Sort by similarity and take top_k
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])][-top_k:][::-1]
        
        return [(filepaths[i], float(similarities[i]), file_types[i]) for i in sorted_indices]

# GUI

class SigLIP2NaFlexApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SigLIP 2 NaFlex Image and Video Search")
        self.geometry("1200x800")
        
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None
        self.db_path = None
        self.search_result_paths = []
        self.query_image_path = None
        
        style = ttk.Style(self)
        style.theme_use('clam')
        
        self.setup_widgets()
    
    def setup_widgets(self):
        """Set up the UI widgets."""
        mainframe = ttk.Frame(self, padding="10")
        mainframe.pack(fill="both", expand=True)
        
        # Controls Panel
        controls_frame = ttk.LabelFrame(mainframe, text="SigLIP 2 NaFlex Controls", padding="10")
        controls_frame.pack(side="left", fill="y", padx=(0, 10))
        
        # Database selection
        db_frame = ttk.LabelFrame(controls_frame, text="Database", padding="5")
        db_frame.pack(fill='x', pady=5)
        self.db_var = tk.StringVar(value="No database selected")
        ttk.Label(db_frame, textvariable=self.db_var, wraplength=250, foreground="blue").pack(anchor='w', pady=2)
        ttk.Button(db_frame, text="Select Database/Folder", command=self.browse_database).pack(fill='x', pady=2)
        
        # Model config
        model_frame = ttk.LabelFrame(controls_frame, text="Model", padding="5")
        model_frame.pack(fill='x', pady=5)
        
        self.model_variant = tk.StringVar(value="google/siglip2-so400m-patch16-naflex")
        
        ttk.Label(model_frame, text="Patches:").pack(anchor='w', pady=(5,0))
        self.max_patches_var = tk.IntVar(value=256)
        patches_frame = ttk.Frame(model_frame)
        patches_frame.pack(fill='x')
        ttk.Spinbox(patches_frame, from_=128, to=1024, increment=128,
                    textvariable=self.max_patches_var, width=10).pack(side='left')
        ttk.Label(patches_frame, text="(256 default, 1024 high res.)").pack(side='left', padx=5)
        
        # Video settings
        video_frame = ttk.LabelFrame(model_frame, text="Video", padding="5")
        video_frame.pack(fill='x', pady=5)
        ttk.Label(video_frame, text="Frames:").pack(anchor='w')
        self.video_frames_var = tk.IntVar(value=5)
        ttk.Spinbox(video_frame, from_=1, to=20, textvariable=self.video_frames_var, width=10).pack(fill='x')
        
        ttk.Label(video_frame, text="Method:").pack(anchor='w', pady=(5,0))
        self.video_method_var = tk.StringVar(value="uniform")
        method_options = ["uniform", "start", "mid", "end"]
        ttk.Combobox(video_frame, textvariable=self.video_method_var, values=method_options, width=15).pack(fill='x')
        
        # Folder
        folder_frame = ttk.LabelFrame(controls_frame, text="Folder", padding="5")
        folder_frame.pack(fill='x', pady=5)
        self.folder_var = tk.StringVar()
        entry_folder = ttk.Entry(folder_frame, textvariable=self.folder_var, width=40)
        entry_folder.pack(fill="x", pady=2)
        ttk.Button(folder_frame, text="Browse...", command=self.browse_folder).pack(fill='x')
        
        # Query
        query_frame = ttk.LabelFrame(controls_frame, text="Search", padding="5")
        query_frame.pack(fill='x', pady=5)
        ttk.Label(query_frame, text="Text:").pack(anchor='w')
        self.query_text_var = tk.StringVar()
        query_text_entry = ttk.Entry(query_frame, textvariable=self.query_text_var)
        query_text_entry.pack(fill='x', pady=(0, 5))
        
        ttk.Label(query_frame, text="Image:").pack(anchor='w', pady=(5,0))
        self.query_image_var = tk.StringVar(value="No query image")
        ttk.Button(query_frame, text="Load Query Image...",
                  command=self.browse_query_image).pack(fill='x', pady=2)
        ttk.Label(query_frame, textvariable=self.query_image_var,
                 wraplength=250, foreground="blue").pack(anchor='w', pady=(5,0))
        
        # Options
        options_frame = ttk.LabelFrame(controls_frame, text="Options", padding="5")
        options_frame.pack(fill='x', pady=5)
        
        device_str, device_msg = setup_device()
        ttk.Label(options_frame, text=f"Device: {device_msg}",
                 foreground="green" if "GPU" in device_msg else "orange").pack(anchor='w')
        self.device_var = tk.StringVar(value=device_str)
        
        ttk.Label(options_frame, text="Results:").pack(anchor='w', pady=(10,0))
        self.top_k_var = tk.IntVar(value=20)
        ttk.Spinbox(options_frame, from_=1, to=100, textvariable=self.top_k_var, width=15).pack(fill='x')
        
        ttk.Label(options_frame, text="Threshold:").pack(anchor='w', pady=(5,0))
        self.threshold_var = tk.DoubleVar(value=0.0)
        threshold_frame = ttk.Frame(options_frame)
        threshold_frame.pack(fill='x')
        ttk.Scale(threshold_frame, from_=0.0, to=1.0, variable=self.threshold_var,
                 orient='horizontal').pack(side='left', fill='x', expand=True)
        self.threshold_label = ttk.Label(threshold_frame, text="0.00")
        self.threshold_label.pack(side='right', padx=5)
        self.threshold_var.trace('w', self.update_threshold_label)
        
        # Actions
        actions_frame = ttk.LabelFrame(controls_frame, text="Actions", padding="5")
        actions_frame.pack(fill='x', pady=10)
        
        self.load_model_button = ttk.Button(actions_frame, text="1. Load Model",
                                            command=self.threaded_load_model)
        self.load_model_button.pack(fill='x', pady=3)
        
        self.index_button = ttk.Button(actions_frame, text="2. Index",
                                       command=self.threaded_index, state="disabled")
        self.index_button.pack(fill='x', pady=3)
        
        self.search_button = ttk.Button(actions_frame, text="3. Search",
                                        command=self.threaded_search, state="disabled")
        self.search_button.pack(fill='x', pady=3)
        
        ttk.Button(actions_frame, text="Cleanup Database",
                  command=self.cleanup_database, state="normal").pack(fill='x', pady=3)
        
        # Status Bar
        status_frame = ttk.Frame(controls_frame)
        status_frame.pack(side="bottom", fill="x", pady=10)
        self.status_var = tk.StringVar(value="Select database and load model")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, wraplength=280)
        status_label.pack(fill="x")
        
        # Results Panel
        results_frame = ttk.LabelFrame(mainframe, text="Search Results", padding="10")
        results_frame.pack(side="left", fill="both", expand=True)
        
        paned_window = ttk.PanedWindow(results_frame, orient='horizontal')
        paned_window.pack(fill='both', expand=True)
        
        list_frame = ttk.Frame(paned_window)
        stats_frame = ttk.Frame(list_frame)
        stats_frame.pack(fill='x', pady=(0,5))
        self.stats_label = ttk.Label(stats_frame, text="No search performed")
        self.stats_label.pack(anchor='w')
        
        self.results_listbox = tk.Listbox(list_frame, height=20, font=('Consolas', 10))
        self.results_listbox.pack(side='left', fill='both', expand=True)
        list_scrollbar = ttk.Scrollbar(list_frame, orient='vertical',
                                      command=self.results_listbox.yview)
        list_scrollbar.pack(side='right', fill='y')
        self.results_listbox.config(yscrollcommand=list_scrollbar.set)
        self.results_listbox.bind('<<ListboxSelect>>', self.on_result_select)
        paned_window.add(list_frame, weight=1)
        
        preview_frame = ttk.Frame(paned_window)
        ttk.Label(preview_frame, text="Preview:", font=('', 10, 'bold')).pack(anchor='w')
        self.image_canvas = tk.Canvas(preview_frame, bg='gray', width=500, height=500)
        self.image_canvas.pack(fill='both', expand=True)
        paned_window.add(preview_frame, weight=2)
    
    def browse_database(self):
        """Browse for a database file or a folder to create one."""
        choice = messagebox.askquestion("Selection", 
            "Do you want to select an existing database?\n\nYes = Select an existing .db file\nNo = Create a new database from a folder")
        
        if choice == 'yes':
            path = filedialog.askopenfilename(
                title="Select existing database",
                filetypes=[("SQLite Database", "*.db"), ("All files", "*.*")]
            )
            if path and os.path.isfile(path):
                self.db_path = path
                self.db_var.set(os.path.basename(path))
                init_db(self.db_path)
                self.update_status(f"Database loaded: {os.path.basename(path)}")
        else:
            path = filedialog.askdirectory(title="Select folder for new database")
            if path and os.path.isdir(path):
                db_filename = os.path.join(path, "siglip2_naflex_embeddings.db")
                self.db_path = db_filename
                self.db_var.set(f"New: {os.path.basename(db_filename)}")
                init_db(self.db_path)
                self.folder_var.set(path)
                self.update_status(f"New database created: {os.path.basename(db_filename)}")
    
    def update_threshold_label(self, *args):
        """Update the threshold value label."""
        self.threshold_label.config(text=f"{self.threshold_var.get():.2f}")
    
    def update_status(self, message, color="black"):
        """Update the status bar message."""
        self.status_var.set(message)
        self.update_idletasks()
    
    def threaded_task(self, target_func, args=()):
        """Run a function in a separate thread to keep the GUI responsive."""
        thread = threading.Thread(target=target_func, args=args)
        thread.daemon = True
        thread.start()
    
    def browse_folder(self):
        """Browse for a folder to index."""
        path = filedialog.askdirectory(title="Select folder")
        if path:
            self.folder_var.set(path)
            self.update_status(f"Folder selected: {path}")
    
    def browse_query_image(self):
        """Browse for an image to use in a search query."""
        path = filedialog.askopenfilename(
            title="Select an image for the search",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif *.webp")]
        )
        if path:
            self.query_image_var.set(os.path.basename(path))
            self.query_image_path = path
            self.update_status(f"Query image: {os.path.basename(path)}")
        else:
            self.query_image_var.set("No query image")
            self.query_image_path = None
    
    def cleanup_database(self):
        """Remove orphaned entries from the database."""
        if not self.db_path:
            messagebox.showerror("Error", "Please select a database first")
            return
        
        if messagebox.askyesno("Confirm", "Do you want to remove entries for deleted files from the database?"):
            self.update_status("Cleaning up database...")
            orphaned_count = cleanup_orphaned_entries(self.db_path, self.update_status)
            messagebox.showinfo("Complete", f"Removed {orphaned_count} orphaned embeddings.")
    
    def threaded_load_model(self):
        """Load the model in a separate thread."""
        self.load_model_button.config(state="disabled")
        self.index_button.config(state="disabled")
        self.search_button.config(state="disabled")
        self.update_status(f"Loading model: {self.model_variant.get()}...")
        self.threaded_task(self.load_model_task)
    
    def load_model_task(self):
        """The actual task of loading the model."""
        try:
            device_choice = self.device_var.get()
            if device_choice == 'cuda':
                self.device = torch.device('cuda')
                self.dtype = torch.float16
            elif device_choice == 'dml':
                self.device = torch_directml.device() if torch_directml else torch.device('cpu')
                self.dtype = torch.float32
            else:
                self.device = torch.device('cpu')
                self.dtype = torch.float32
            
            model_name = self.model_variant.get()
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Siglip2Model.from_pretrained(
                model_name,
                torch_dtype=self.dtype
            ).to(self.device)
            self.model.eval()
            
            if device_choice != 'dml' and hasattr(torch, 'compile'):
                self.update_status("Optimizing model...")
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception:
                    pass # Compilation can fail on some setups
            
            self.after(0, self.load_model_finished)
        
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.after(0, lambda: messagebox.showerror("Model Error", error_msg))
            self.after(0, lambda: self.load_model_button.config(state="normal"))
            self.after(0, lambda: self.update_status("Error loading model."))
    
    def load_model_finished(self):
        """Callback function after the model is loaded."""
        device_type = str(self.device).upper() if self.device.type == 'cuda' else str(self.device)
        self.update_status(f"Model loaded on {device_type}. Ready!")
        self.index_button.config(state="normal")
        self.search_button.config(state="normal")
        self.load_model_button.config(state="normal", text="Model Loaded")
    
    def threaded_index(self):
        """Start the indexing process in a separate thread."""
        if not self.db_path:
            messagebox.showerror("Error", "Please select a database first")
            return
        
        folder_path = self.folder_var.get()
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror("Error", "Please select a valid folder.")
            return
        
        self.index_button.config(state="disabled")
        self.search_button.config(state="disabled")
        self.update_status("Indexing in progress...")
        self.threaded_task(self.index_task, (folder_path,))
    
    def index_task(self, folder_path):
        """The actual task of indexing files."""
        try:
            max_patches = self.max_patches_var.get()
            video_frames = self.video_frames_var.get()
            video_method = self.video_method_var.get()
            
            index_images_and_videos(
                folder_path,
                self.device,
                self.processor,
                self.model,
                self.db_path,
                batch_size=8,
                progress_callback=self.update_status,
                max_num_patches=max_patches,
                video_frames=video_frames,
                video_method=video_method
            )
            
            self.after(0, self.index_finished)
        
        except Exception as e:
            error_msg = f"Error during indexing: {str(e)}"
            self.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.after(0, lambda: self.index_button.config(state="normal"))
            self.after(0, lambda: self.search_button.config(state="normal"))
    
    def index_finished(self):
        """Callback function after indexing is complete."""
        self.update_status("Indexing complete!")
        self.index_button.config(state="normal", text="Indexed")
        self.search_button.config(state="normal")
        messagebox.showinfo("Complete", "Indexing has finished.")
    
    def threaded_search(self):
        """Start the search process in a separate thread."""
        if not self.db_path:
            messagebox.showerror("Error", "Please select a database first")
            return
        
        if not self.query_text_var.get() and not self.query_image_path:
            messagebox.showwarning("Warning", "Please enter text or select an image to search.")
            return
        
        self.search_button.config(state="disabled")
        self.update_status("Searching...")
        self.threaded_task(self.search_task)
    
    def search_task(self):
        """The actual task of searching the database."""
        try:
            max_patches = self.max_patches_var.get()
            
            query_embedding = get_query_embedding(
                self.query_text_var.get(),
                self.query_image_path,
                self.device,
                self.processor,
                self.model,
                max_num_patches=max_patches
            )
            
            if query_embedding is None:
                self.after(0, lambda: messagebox.showerror("Error", "Could not generate query embedding."))
                self.after(0, lambda: self.search_button.config(state="normal"))
                return
            
            results = search_db(
                query_embedding,
                self.db_path,
                self.top_k_var.get(),
                self.threshold_var.get()
            )
            
            self.after(0, self.search_finished, results)
        
        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            self.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.after(0, lambda: self.search_button.config(state="normal"))
    
    def search_finished(self, results):
        """Callback function after the search is complete to display results."""
        self.results_listbox.delete(0, tk.END)
        self.search_result_paths.clear()
        
        if not results:
            self.update_status("No results found.")
            self.stats_label.config(text="No results")
            self.search_button.config(state="normal")
            return
        
        scores = [score for _, score, _ in results]
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        for i, (path, score, file_type) in enumerate(results, 1):
            filename = os.path.basename(path)
            type_icon = "[V]" if file_type == 'video' else "[I]"
            
            if score >= 0.8:
                indicator = "*"
            elif score >= 0.6:
                indicator = "-"
            else:
                indicator = "."
            
            self.results_listbox.insert(
                tk.END,
                f"{indicator}{type_icon} {i:3d}. {filename[:35]:<35} | Score: {score:.4f}"
            )
            self.search_result_paths.append((path, file_type))
        
        self.stats_label.config(
            text=f"Found {len(results)} results | Max: {max_score:.3f} | Avg: {avg_score:.3f} | Min: {min_score:.3f}"
        )
        
        self.update_status(f"Found {len(results)} results.")
        self.search_button.config(state="normal")
        
        if self.search_result_paths:
            self.results_listbox.selection_set(0)
            self.on_result_select(None)
    
    def on_result_select(self, event):
        """Handle selection of an item in the results list."""
        selection_indices = self.results_listbox.curselection()
        if not selection_indices:
            return
        
        selected_index = selection_indices[0]
        path, file_type = self.search_result_paths[selected_index]
        
        if file_type == 'image':
            self.display_image(path)
        elif file_type == 'video':
            self.display_video_thumbnail(path)
    
    def display_image(self, path):
        """Display an image preview in the canvas."""
        try:
            img = Image.open(path)
            canvas_w = self.image_canvas.winfo_width()
            canvas_h = self.image_canvas.winfo_height()
            
            if canvas_w <= 1 or canvas_h <= 1: # Canvas not yet rendered
                canvas_w, canvas_h = 500, 500
            
            img_w, img_h = img.size
            scale = min((canvas_w - 20) / img_w, (canvas_h - 20) / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(img)
            
            self.image_canvas.delete("all")
            self.image_canvas.create_image(canvas_w//2, canvas_h//2, image=self.tk_image)
            
            file_size = os.path.getsize(path) / (1024 * 1024)
            info_text = f"{os.path.basename(path)} | {img_w}x{img_h} | {file_size:.1f} MB"
            self.image_canvas.create_text(
                canvas_w//2, canvas_h - 10,
                text=info_text,
                fill="white",
                font=('Arial', 9),
                anchor="s"
            )
        
        except Exception as e:
            self.update_status(f"Display error: {e}")
    
    def display_video_thumbnail(self, path):
        """Display a video thumbnail in the canvas."""
        try:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self.update_status("Could not load video thumbnail")
                return
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            canvas_w = self.image_canvas.winfo_width()
            canvas_h = self.image_canvas.winfo_height()
            
            if canvas_w <= 1 or canvas_h <= 1: # Canvas not yet rendered
                canvas_w, canvas_h = 500, 500
            
            img_w, img_h = img.size
            scale = min((canvas_w - 20) / img_w, (canvas_h - 20) / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(img)
            
            self.image_canvas.delete("all")
            self.image_canvas.create_image(canvas_w//2, canvas_h//2, image=self.tk_image)
            
            file_size = os.path.getsize(path) / (1024 * 1024)
            info_text = f"[VIDEO] {os.path.basename(path)} | {file_size:.1f} MB"
            self.image_canvas.create_text(
                canvas_w//2, canvas_h - 10,
                text=info_text,
                fill="yellow",
                font=('Arial', 10, 'bold'),
                anchor="s"
            )
        
        except Exception as e:
            self.update_status(f"Video display error: {e}")

# CLI

def cli_mode():
    """Command-Line Interface mode."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SigLIP 2 NaFlex Image/Video Search')
    parser.add_argument('--index', type=str, help='Path to the folder to index')
    parser.add_argument('--search-text', type=str, help='Text to search for')
    parser.add_argument('--search-image', type=str, help='Image to search with')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results to return')
    parser.add_argument('--db', type=str, default='siglip2_naflex_embeddings.db', help='Database file path')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'dml'], help='Force a specific device')
    parser.add_argument('--model', type=str, default='google/siglip2-so400m-patch16-naflex', help='Model to use')
    parser.add_argument('--max-patches', type=int, default=256, help='Max patches for the model')
    parser.add_argument('--video-frames', type=int, default=5, help='Number of frames to extract per video')
    parser.add_argument('--video-method', type=str, default='uniform', choices=['uniform', 'start', 'mid', 'end'])
    parser.add_argument('--cleanup', action='store_true', help='Clean up orphaned embeddings from the database')
    
    args = parser.parse_args()
    
    if args.device:
        device_str, msg = setup_device(args.device)
    else:
        device_str, msg = setup_device()
    print(f"Device: {msg}")
    
    if device_str == 'cuda':
        device = torch.device('cuda')
        dtype = torch.float16
    elif device_str == 'dml' and torch_directml:
        device = torch_directml.device()
        dtype = torch.float32
    else:
        device = torch.device('cpu')
        dtype = torch.float32
    
    init_db(args.db)
    
    if args.cleanup:
        print("Cleaning up orphaned embeddings...")
        count = cleanup_orphaned_entries(args.db)
        print(f"Removed {count} orphaned embeddings.")
    
    print(f"Loading model {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Siglip2Model.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    print("Model loaded!")
    
    if args.index:
        if os.path.isdir(args.index):
            print(f"\nIndexing folder: {args.index}")
            index_images_and_videos(
                args.index,
                device,
                processor,
                model,
                args.db,
                batch_size=8,
                max_num_patches=args.max_patches,
                video_frames=args.video_frames,
                video_method=args.video_method
            )
            print("Indexing complete!")
        else:
            print(f"Error: {args.index} is not a valid folder.")
    
    if args.search_text or args.search_image:
        print("\n" + "="*60)
        print("PERFORMING SEARCH")
        print("="*60)
        
        if args.search_text:
            print(f"Text query: '{args.search_text}'")
        if args.search_image:
            print(f"Image query: {args.search_image}")
        
        query_embedding = get_query_embedding(
            args.search_text,
            args.search_image,
            device,
            processor,
            model,
            max_num_patches=args.max_patches
        )
        
        if query_embedding is None:
            print("Error: could not generate query embedding.")
            return
        
        results = search_db(query_embedding, args.db, args.top_k)
        
        if not results:
            print("\nNo results found.")
        else:
            print(f"\n--- TOP {len(results)} RESULTS ---")
            print("-" * 70)
            for i, (path, score, file_type) in enumerate(results, 1):
                filename = os.path.basename(path)
                type_icon = "[V]" if file_type == 'video' else "[I]"
                print(f"{i:3d}. {type_icon} Score: {score:.4f} | {filename}")
                if i <= 3: # full path top results
                    print(f"     Path: {path}")
            print("-" * 70)
            
            scores = [s for _, s, _ in results]
            print(f"\nStatistics:")
            print(f" Max score: {max(scores):.4f}")
            print(f" Avg score: {np.mean(scores):.4f}")
            print(f" Min score: {min(scores):.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cli_mode()
    else:
        print("Starting SigLIP 2 NaFlex GUI...")
        app = SigLIP2NaFlexApp()
        app.mainloop()