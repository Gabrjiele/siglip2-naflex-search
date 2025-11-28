import argparse
import json
import os
import sqlite3
import subprocess
import sys
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import filedialog, messagebox, ttk, simpledialog
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from tqdm import tqdm
from transformers import AutoProcessor, Siglip2Model
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
try:
    import torch_directml
except ImportError:
    torch_directml = None
CONFIG_FILE = Path(__file__).parent / 'siglip2_config.json'
DEFAULT_MODEL = 'google/siglip2-so400m-patch16-naflex'
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
DB_SCHEMA = f"\nCREATE TABLE IF NOT EXISTS embeddings (\n    filepath TEXT PRIMARY KEY,\n    modified_at REAL NOT NULL,\n    embedding BLOB NOT NULL,\n    model_version TEXT DEFAULT '{DEFAULT_MODEL}',\n    file_type TEXT DEFAULT 'image'\n);\n"

def load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f'Error loading config file: {e}')
    return {}

def save_config(config: Dict[str, Any]) -> None:
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        print(f'Error saving config file: {e}')

def setup_device(forced_device: Optional[str]=None) -> Tuple[str, str]:
    if forced_device:
        if forced_device == 'dml' and (torch_directml is None or not torch_directml.is_available()):
            return ('cpu', 'DirectML not available. Falling back to CPU.')
        if forced_device == 'cuda' and (not torch.cuda.is_available()):
            return ('cpu', 'CUDA not available. Falling back to CPU.')
        return (forced_device, f'Device forced to {forced_device.upper()}.')
    if torch.cuda.is_available():
        return ('cuda', 'NVIDIA GPU (CUDA) detected.')
    if torch_directml and torch_directml.is_available():
        return ('dml', 'AMD/Intel GPU (DirectML) detected.')
    return ('cpu', 'No compatible GPU found. Using CPU.')

def normalize_embedding(features: torch.Tensor) -> torch.Tensor:
    return features / torch.linalg.norm(features, ord=2, dim=-1, keepdim=True)

def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(DB_SCHEMA)

def cleanup_orphaned_entries(db_path: str, progress_callback: Optional[Callable]=None) -> int:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT filepath FROM embeddings')
        all_paths = cursor.fetchall()
        if progress_callback:
            progress_callback(f'Checking {len(all_paths)} files in the database...')
        orphaned_paths = [path for path, in all_paths if not os.path.exists(path)]
        if orphaned_paths:
            if progress_callback:
                progress_callback(f'Removing {len(orphaned_paths)} orphaned embeddings...')
            cursor.executemany('DELETE FROM embeddings WHERE filepath=?', [(p,) for p in orphaned_paths])
            conn.commit()
            if progress_callback:
                progress_callback(f'Cleanup complete: removed {len(orphaned_paths)} orphaned embeddings.')
        elif progress_callback:
            progress_callback('No orphaned embeddings found.')
        return len(orphaned_paths)

def extract_video_frames(video_path: str, num_frames: int=5) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error opening video: {video_path}')
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def index_files(folder_path: str, device: torch.device, processor, model, db_path: str, batch_size: int=10, progress_callback: Optional[Callable]=None, max_num_patches: int=256, video_frames: int=5) -> None:
    if progress_callback:
        progress_callback('Cleaning database of deleted files...')
    cleanup_orphaned_entries(db_path, progress_callback)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    all_files = []
    for ext in IMAGE_EXTENSIONS:
        all_files.extend(Path(folder_path).rglob(f'*{ext}'))
    for ext in VIDEO_EXTENSIONS:
        all_files.extend(Path(folder_path).rglob(f'*{ext}'))
    paths_to_process = []
    if progress_callback:
        progress_callback(f'Checking {len(all_files)} files...')
    for path_obj in tqdm(all_files, desc='Checking file modification times'):
        path = str(path_obj)
        try:
            last_modified = os.path.getmtime(path)
            cursor.execute('SELECT modified_at FROM embeddings WHERE filepath=?', (path,))
            result = cursor.fetchone()
            if not result or result[0] < last_modified:
                paths_to_process.append(path)
        except FileNotFoundError:
            continue
    if not paths_to_process:
        if progress_callback:
            progress_callback('Database is up to date.')
        conn.close()
        return
    images_to_process = [p for p in paths_to_process if p.lower().endswith(IMAGE_EXTENSIONS)]
    videos_to_process = [p for p in paths_to_process if p.lower().endswith(VIDEO_EXTENSIONS)]
    if progress_callback:
        progress_callback(f'Indexing {len(images_to_process)} images and {len(videos_to_process)} videos...')
    processed_count = 0
    total_to_process = len(paths_to_process)
    if images_to_process:
        for i in tqdm(range(0, len(images_to_process), batch_size), desc='Processing image batches'):
            batch_paths = images_to_process[i:i + batch_size]
            batch_images, valid_paths, mtimes = ([], [], [])
            for path in batch_paths:
                try:
                    batch_images.append(Image.open(path).convert('RGB'))
                    valid_paths.append(path)
                    mtimes.append(os.path.getmtime(path))
                except Exception as e:
                    print(f'Error loading image {path}: {e}')
            if not batch_images:
                continue
            try:
                inputs = processor(images=batch_images, return_tensors='pt', max_num_patches=max_num_patches).to(device)
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    image_features = normalize_embedding(image_features)
                for idx, (path, mtime) in enumerate(zip(valid_paths, mtimes)):
                    embedding = image_features[idx].cpu().numpy().astype(np.float32).tobytes()
                    cursor.execute('REPLACE INTO embeddings (filepath, modified_at, embedding, file_type) VALUES (?, ?, ?, ?)', (path, mtime, embedding, 'image'))
                processed_count += len(valid_paths)
                if progress_callback:
                    progress_callback(f'Indexing: {processed_count}/{total_to_process}')
            except Exception as e:
                print(f'Error processing image batch: {e}')
            conn.commit()
    for path in tqdm(videos_to_process, desc='Processing videos'):
        try:
            frames = extract_video_frames(path, num_frames=video_frames)
            if not frames:
                continue
            frame_pils = [Image.fromarray(frame) for frame in frames]
            inputs = processor(images=frame_pils, return_tensors='pt', max_num_patches=max_num_patches).to(device)
            with torch.no_grad():
                frame_features = model.get_image_features(**inputs)
                frame_features = normalize_embedding(frame_features)
                frame_embeddings = frame_features.cpu().numpy()
            video_embedding = np.mean(frame_embeddings, axis=0).astype(np.float32)
            embedding_bytes = video_embedding.tobytes()
            last_modified = os.path.getmtime(path)
            cursor.execute('REPLACE INTO embeddings (filepath, modified_at, embedding, file_type) VALUES (?, ?, ?, ?)', (path, last_modified, embedding_bytes, 'video'))
            processed_count += 1
            if progress_callback:
                progress_callback(f'Indexing: {processed_count}/{total_to_process}')
        except Exception as e:
            print(f'Error processing video {path}: {e}')
        conn.commit()
    conn.close()
    if progress_callback:
        progress_callback(f'Indexing complete. Processed {total_to_process} new/modified files.')

def get_query_embedding(query_text: str, query_image_path: Optional[str], device: torch.device, processor, model, max_num_patches: int=256) -> Optional[np.ndarray]:
    text_embedding, image_embedding = (None, None)
    with torch.no_grad():
        if query_text:
            inputs = processor(text=[query_text.lower()], return_tensors='pt', padding='max_length', max_length=64).to(device)
            text_features = model.get_text_features(**inputs)
            text_embedding = normalize_embedding(text_features).cpu().numpy().astype(np.float32)
        if query_image_path:
            try:
                image = Image.open(query_image_path).convert('RGB')
                inputs = processor(images=image, return_tensors='pt', max_num_patches=max_num_patches).to(device)
                image_features = model.get_image_features(**inputs)
                image_embedding = normalize_embedding(image_features).cpu().numpy().astype(np.float32)
            except Exception as e:
                print(f'Error processing query image: {e}')
                return None
    if text_embedding is not None and image_embedding is not None:
        return (text_embedding + image_embedding) / 2
    return text_embedding if text_embedding is not None else image_embedding

def search_db(query_embedding: np.ndarray, db_path: str, top_k: int=10, similarity_threshold: float=-1.0, batch_size: int=1000) -> List[Tuple[str, float, str]]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT filepath, embedding, file_type FROM embeddings')
        results = []
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            filepaths = [row[0] for row in batch]
            file_types = [row[2] for row in batch]
            db_embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in batch])
            if db_embeddings.ndim == 1:
                db_embeddings = db_embeddings.reshape(1, -1)
            similarities = np.dot(db_embeddings, query_embedding.T).squeeze()
            for i, sim in enumerate(similarities):
                if sim >= similarity_threshold:
                    results.append((filepaths[i], float(sim), file_types[i]))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

class SigLIP2NaFlexApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('SigLIP 2 NaFlex Image and Video Search')
        self.geometry('1200x800')
        self.model: Optional[Siglip2Model] = None
        self.processor = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self.db_path: Optional[str] = None
        self.query_image_path: Optional[str] = None
        self.current_display_path: Optional[str] = None
        self.search_results: List[Tuple[str, float, str, Optional[float]]] = []
        self.canvas_scale = 1.0
        self.canvas_offset_x, self.canvas_offset_y = (0, 0)
        self.drag_start_x, self.drag_start_y = (0, 0)
        self.original_image: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.zoom_timer: Optional[str] = None
        self.config = load_config()
        ttk.Style(self).theme_use('clam')
        self.setup_widgets()
        self.load_saved_paths()
        self.after(100, self.threaded_load_model)

    def setup_widgets(self):
        mainframe = ttk.Frame(self, padding='10')
        mainframe.pack(fill='both', expand=True)
        main_paned = ttk.PanedWindow(mainframe, orient='horizontal')
        main_paned.pack(fill='both', expand=True)
        controls_frame = ttk.LabelFrame(main_paned, text='Controls', padding='10')
        main_paned.add(controls_frame, weight=0)
        db_frame = ttk.LabelFrame(controls_frame, text='Database', padding=5)
        db_frame.pack(fill='x', pady=5)
        self.db_var = tk.StringVar(value='No database selected')
        ttk.Label(db_frame, textvariable=self.db_var, wraplength=250).pack(anchor='w')
        db_btn_frame = ttk.Frame(db_frame)
        db_btn_frame.pack(fill='x', pady=2)
        ttk.Button(db_btn_frame, text='Open Existing...', command=self.browse_existing_database).pack(side='left', expand=True, fill='x', padx=(0, 2))
        ttk.Button(db_btn_frame, text='Create New...', command=self.browse_database).pack(side='left', expand=True, fill='x', padx=(2, 0))
        folder_frame = ttk.LabelFrame(controls_frame, text='Folder to Index', padding=5)
        folder_frame.pack(fill='x', pady=5)
        self.folder_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.folder_var).pack(fill='x', pady=2)
        ttk.Button(folder_frame, text='Browse...', command=self.browse_folder).pack(fill='x')
        query_frame = ttk.LabelFrame(controls_frame, text='Search Query', padding=5)
        query_frame.pack(fill='x', pady=5)
        ttk.Label(query_frame, text='Text:').pack(anchor='w')
        self.query_text_var = tk.StringVar()
        ttk.Entry(query_frame, textvariable=self.query_text_var).pack(fill='x', pady=(0, 5))
        ttk.Label(query_frame, text='Image:').pack(anchor='w')
        self.query_image_var = tk.StringVar(value='No query image')
        ttk.Label(query_frame, textvariable=self.query_image_var, wraplength=250).pack(anchor='w')
        btn_frame = ttk.Frame(query_frame)
        btn_frame.pack(fill='x', pady=2)
        ttk.Button(btn_frame, text='Load...', command=self.browse_query_image).pack(side='left', expand=True, fill='x')
        ttk.Button(btn_frame, text='Clear', command=self.clear_query_image).pack(side='left', expand=True, fill='x')
        options_frame = ttk.LabelFrame(controls_frame, text='Options', padding=5)
        options_frame.pack(fill='x', pady=5)
        device_str, device_msg = setup_device()
        ttk.Label(options_frame, text=f'Device: {device_msg}').pack(anchor='w')
        self.device_var = tk.StringVar(value=device_str)
        ttk.Label(options_frame, text='Max Patches:').pack(anchor='w', pady=(5, 0))
        self.max_patches_var = tk.IntVar(value=256)
        ttk.Spinbox(options_frame, from_=128, to=1024, increment=128, textvariable=self.max_patches_var).pack(fill='x')
        ttk.Label(options_frame, text='Results:').pack(anchor='w', pady=(5, 0))
        self.top_k_var = tk.IntVar(value=20)
        ttk.Spinbox(options_frame, from_=1, to=100, textvariable=self.top_k_var).pack(fill='x')
        actions_frame = ttk.LabelFrame(controls_frame, text='Actions', padding=5)
        actions_frame.pack(fill='x', pady=10)
        self.load_model_button = ttk.Button(actions_frame, text='1. Load Model', command=self.threaded_load_model)
        self.load_model_button.pack(fill='x', pady=3)
        self.index_button = ttk.Button(actions_frame, text='2. Index Folder', command=self.threaded_index, state='disabled')
        self.index_button.pack(fill='x', pady=3)
        self.search_button = ttk.Button(actions_frame, text='3. Search', command=self.threaded_search, state='disabled')
        self.search_button.pack(fill='x', pady=3)
        ttk.Button(actions_frame, text='Cleanup Database', command=self.cleanup_database).pack(fill='x', pady=3)
        self.status_var = tk.StringVar(value='Select database and load model')
        ttk.Label(controls_frame, textvariable=self.status_var, wraplength=280).pack(side='bottom', fill='x', pady=10)
        results_frame = ttk.LabelFrame(main_paned, text='Results', padding='10')
        main_paned.add(results_frame, weight=1)
        paned_window = ttk.PanedWindow(results_frame, orient='horizontal')
        paned_window.pack(fill='both', expand=True)
        list_frame = ttk.Frame(paned_window)
        self.stats_label = ttk.Label(list_frame, text='No search performed')
        self.stats_label.pack(anchor='w')
        rescore_frame = ttk.Frame(list_frame)
        rescore_frame.pack(fill='x', pady=5)
        self.rescore_button = ttk.Button(rescore_frame, text='Rescore...', command=self.open_rescore_dialog, state='disabled')
        self.rescore_button.pack(side='left')
        self.clear_rescore_button = ttk.Button(rescore_frame, text='Clear Rescore', command=self.clear_rescore, state='disabled')
        self.clear_rescore_button.pack(side='left', padx=5)
        self.results_listbox = tk.Listbox(list_frame, font=('Consolas', 10))
        self.results_listbox.pack(side='left', fill='both', expand=True)
        list_scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.results_listbox.yview)
        list_scrollbar.pack(side='right', fill='y')
        self.results_listbox.config(yscrollcommand=list_scrollbar.set)
        self.results_listbox.bind('<<ListboxSelect>>', self.on_result_select)
        paned_window.add(list_frame, weight=1)
        preview_frame = ttk.Frame(paned_window)
        ttk.Label(preview_frame, text='Preview (Drag to pan, Wheel to zoom, Double-click to open)', font=('', 8)).pack()
        self.image_canvas = tk.Canvas(preview_frame, bg='gray')
        self.image_canvas.pack(fill='both', expand=True)
        self.image_canvas.bind('<ButtonPress-1>', self.on_canvas_click)
        self.image_canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.image_canvas.bind('<MouseWheel>', self.on_canvas_zoom)
        self.image_canvas.bind('<Double-Button-1>', self.on_canvas_double_click)
        self.image_canvas.bind('<Button-3>', self.on_canvas_right_click)
        paned_window.add(preview_frame, weight=2)

    def threaded_task(self, target_func: Callable, *args):
        thread = threading.Thread(target=target_func, args=args, daemon=True)
        thread.start()

    def update_status(self, message: str):
        self.status_var.set(message)
        self.update_idletasks()

    def load_saved_paths(self):
        if 'db_path' in self.config and os.path.exists(self.config['db_path']):
            self.db_path = self.config['db_path']
            self.db_var.set(os.path.basename(self.db_path))
            init_db(self.db_path)
            self.update_status(f'Loaded saved database: {os.path.basename(self.db_path)}')
        if 'folder_path' in self.config and os.path.exists(self.config['folder_path']):
            self.folder_var.set(self.config['folder_path'])

    def browse_database(self):
        path = filedialog.asksaveasfilename(title='Create New Database File', initialdir=self.config.get('db_path', ''), filetypes=[('SQLite Database', '*.db')], defaultextension='.db')
        if path:
            self.db_path = path
            self.db_var.set(os.path.basename(path))
            init_db(self.db_path)
            self.config['db_path'] = path
            save_config(self.config)
            self.update_status(f'Database set to: {os.path.basename(path)}')
            if not Path(path).parent.is_dir():
                return
            self.folder_var.set(str(Path(path).parent))
            self.config['folder_path'] = str(Path(path).parent)
            save_config(self.config)

    def browse_existing_database(self):
        path = filedialog.askopenfilename(title='Select Existing Database File', initialdir=self.config.get('db_path', ''), filetypes=[('SQLite Database', '*.db')])
        if path:
            self.db_path = path
            self.db_var.set(os.path.basename(path))
            init_db(self.db_path)
            self.config['db_path'] = path
            save_config(self.config)
            self.update_status(f'Database set to: {os.path.basename(path)}')

    def browse_folder(self):
        path = filedialog.askdirectory(title='Select Folder to Index', initialdir=self.config.get('folder_path', ''))
        if path:
            self.folder_var.set(path)
            self.config['folder_path'] = path
            save_config(self.config)

    def browse_query_image(self):
        path = filedialog.askopenfilename(filetypes=[('Images', ' '.join((f'*{ext}' for ext in IMAGE_EXTENSIONS)))])
        if path:
            self.query_image_path = path
            self.query_image_var.set(os.path.basename(path))

    def clear_query_image(self):
        self.query_image_path = None
        self.query_image_var.set('No query image')

    def cleanup_database(self):
        if not self.db_path:
            messagebox.showerror('Error', 'Please select a database first.')
            return
        if messagebox.askyesno('Confirm', 'Remove entries for deleted files from the database?'):
            self.threaded_task(self._cleanup_task)

    def _cleanup_task(self):
        self.update_status('Cleaning up database...')
        count = cleanup_orphaned_entries(self.db_path, self.update_status)
        self.after(0, lambda: messagebox.showinfo('Complete', f'Removed {count} orphaned embeddings.'))
        self.update_status('Cleanup complete.')

    def threaded_load_model(self):
        self.load_model_button.config(state='disabled')
        self.index_button.config(state='disabled')
        self.search_button.config(state='disabled')
        self.update_status(f'Loading model: {DEFAULT_MODEL}...')
        self.threaded_task(self.load_model_task)

    def load_model_task(self):
        try:
            device_choice = self.device_var.get()
            if device_choice == 'cuda':
                self.device = torch.device('cuda')
                self.dtype = torch.float16
            elif device_choice == 'dml' and torch_directml:
                self.device = torch_directml.device()
                self.dtype = torch.float32
            else:
                self.device = torch.device('cpu')
                self.dtype = torch.float32
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(DEFAULT_MODEL)
            attn_implementation = 'sdpa' if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else 'eager'
            self.model = Siglip2Model.from_pretrained(DEFAULT_MODEL, torch_dtype=self.dtype, attn_implementation=attn_implementation).to(self.device)
            self.model.eval()
            self.after(0, self.on_model_load_finished)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Model Error', f'Failed to load model: {e}'))
            self.after(0, lambda: self.load_model_button.config(state='normal'))
            self.after(0, lambda: self.update_status('Error loading model.'))

    def on_model_load_finished(self):
        self.update_status(f'Model loaded on {str(self.device).upper()}. Ready!')
        self.index_button.config(state='normal')
        self.search_button.config(state='normal')
        self.load_model_button.config(state='normal', text='Reload Model')

    def threaded_index(self):
        if not self.db_path or not self.folder_var.get():
            messagebox.showerror('Error', 'Please select a database and a folder to index.')
            return
        self.index_button.config(state='disabled')
        self.search_button.config(state='disabled')
        self.update_status('Indexing in progress...')
        self.threaded_task(self.index_task, self.folder_var.get())

    def index_task(self, folder_path: str):
        try:
            index_files(folder_path, self.device, self.processor, self.model, self.db_path, batch_size=8, progress_callback=self.update_status, max_num_patches=self.max_patches_var.get())
            self.after(0, self.on_index_finished)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Indexing Error', str(e)))
        finally:
            self.after(0, lambda: self.index_button.config(state='normal'))
            self.after(0, lambda: self.search_button.config(state='normal'))

    def on_index_finished(self):
        self.update_status('Indexing complete!')
        messagebox.showinfo('Complete', 'Indexing has finished.')

    def threaded_search(self):
        if not self.db_path:
            messagebox.showerror('Error', 'Please select a database first.')
            return
        if not self.query_text_var.get() and (not self.query_image_path):
            messagebox.showwarning('Warning', 'Please enter text or select an image to search.')
            return
        self.search_button.config(state='disabled')
        self.update_status('Searching...')
        self.threaded_task(self.search_task)

    def search_task(self):
        try:
            query_embedding = get_query_embedding(self.query_text_var.get(), self.query_image_path, self.device, self.processor, self.model, self.max_patches_var.get())
            if query_embedding is None:
                raise ValueError('Could not generate query embedding.')
            results = search_db(query_embedding, self.db_path, self.top_k_var.get())
            self.after(0, self.on_search_finished, results)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Search Error', str(e)))
        finally:
            self.after(0, lambda: self.search_button.config(state='normal'))

    def on_search_finished(self, results: List[Tuple[str, float, str]]):
        self.search_results = [(path, score, ftype, None) for path, score, ftype in results]
        self._update_listbox()
        self.update_status(f'Found {len(results)} results.')
        self.rescore_button.config(state='normal' if results else 'disabled')
        self.clear_rescore_button.config(state='disabled')

    def _update_listbox(self):
        self.results_listbox.delete(0, tk.END)
        if not self.search_results:
            self.stats_label.config(text='No results found.')
            return
        has_rescore = self.search_results and self.search_results[0][3] is not None
        sort_key = lambda x: x[3] if has_rescore else x[1]
        self.search_results.sort(key=sort_key, reverse=True)
        for i, (path, score, ftype, rescore) in enumerate(self.search_results, 1):
            filename = os.path.basename(path)
            type_icon = '[V]' if ftype == 'video' else '[I]'
            display_text = f'{type_icon} {i:2d}. {filename[:35]:<35} | Score: {score:.4f}'
            if rescore is not None:
                display_text += f' | Rescore: {rescore:.4f}'
            self.results_listbox.insert(tk.END, display_text)
        scores = [rescore if has_rescore and rescore is not None else score for _, score, _, rescore in self.search_results]
        stats_text = f'Found {len(scores)} results | Max: {max(scores):.3f} | Avg: {np.mean(scores):.3f}'
        self.stats_label.config(text=stats_text)
        if self.search_results:
            self.results_listbox.selection_set(0)
            self.on_result_select(None)

    def open_rescore_dialog(self):
        query_text = simpledialog.askstring('Rescore', 'Enter new text query to rescore results:', parent=self)
        if query_text:
            self.threaded_task(self.rescore_task, query_text)

    def rescore_task(self, query_text: str):
        self.update_status(f"Rescoring with: '{query_text}'...")
        try:
            rescore_embedding = get_query_embedding(query_text, None, self.device, self.processor, self.model)
            if rescore_embedding is None:
                raise ValueError('Could not generate rescore embedding.')
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for i, (path, score, ftype, _) in enumerate(self.search_results):
                    cursor.execute('SELECT embedding FROM embeddings WHERE filepath=?', (path,))
                    result = cursor.fetchone()
                    if result:
                        embedding = np.frombuffer(result[0], dtype=np.float32)
                        similarity = np.dot(embedding, rescore_embedding.T).squeeze()
                        self.search_results[i] = (path, score, ftype, float(similarity))
            self.after(0, self.on_rescore_finished)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Rescore Error', str(e)))

    def on_rescore_finished(self):
        self._update_listbox()
        self.update_status('Rescore complete.')
        self.clear_rescore_button.config(state='normal')

    def clear_rescore(self):
        self.search_results = [(path, score, ftype, None) for path, score, ftype, _ in self.search_results]
        self._update_listbox()
        self.update_status('Rescore cleared.')
        self.clear_rescore_button.config(state='disabled')

    def on_result_select(self, event: Optional[tk.Event]):
        indices = self.results_listbox.curselection()
        if not indices:
            return
        path, _, file_type, _ = self.search_results[indices[0]]
        self.current_display_path = path
        if file_type == 'image':
            self.display_media(path, is_video=False)
        else:
            self.display_media(path, is_video=True)

    def display_media(self, path: str, is_video: bool):
        try:
            self._reset_pan_zoom()
            self.display_image = None
            if is_video:
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise IOError('Could not read video frame.')
                self.original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                self.original_image = Image.open(path).convert('RGB')
            canvas_w = self.image_canvas.winfo_width()
            canvas_h = self.image_canvas.winfo_height()
            if self.original_image and canvas_w > 1 and (canvas_h > 1):
                img_w, img_h = self.original_image.size
                w_ratio = canvas_w / img_w
                h_ratio = canvas_h / img_h
                self.canvas_scale = min(w_ratio, h_ratio)
            self._render_image_on_canvas()
        except Exception as e:
            self.update_status(f'Display error: {e}')
            self.image_canvas.delete('all')

    def _render_image_on_canvas(self, use_fast_quality: bool=False):
        if not self.original_image:
            self.image_canvas.delete('all')
            return
        canvas_w = self.image_canvas.winfo_width()
        canvas_h = self.image_canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            self.after(100, self._render_image_on_canvas)
            return
        new_w = int(self.original_image.width * self.canvas_scale)
        new_h = int(self.original_image.height * self.canvas_scale)
        if new_w < 1 or new_h < 1:
            self.image_canvas.delete('all')
            return
        if self.display_image is None or self.display_image.size != (new_w, new_h):
            resample_method = Image.Resampling.BILINEAR if use_fast_quality else Image.Resampling.LANCZOS
            self.display_image = self.original_image.resize((new_w, new_h), resample_method)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.image_canvas.delete('all')
        draw_x = canvas_w / 2 + self.canvas_offset_x
        draw_y = canvas_h / 2 + self.canvas_offset_y
        self.image_canvas.create_image(draw_x, draw_y, image=self.tk_image)

    def _reset_pan_zoom(self):
        self.canvas_scale = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

    def on_canvas_click(self, event: tk.Event):
        self.drag_start_x, self.drag_start_y = (event.x, event.y)

    def on_canvas_drag(self, event: tk.Event):
        self.canvas_offset_x += event.x - self.drag_start_x
        self.canvas_offset_y += event.y - self.drag_start_y
        self.drag_start_x, self.drag_start_y = (event.x, event.y)
        self._render_image_on_canvas(use_fast_quality=True)

    def on_canvas_zoom(self, event: tk.Event):
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        self.canvas_scale = max(0.1, min(10.0, self.canvas_scale * factor))
        self._render_image_on_canvas(use_fast_quality=True)
        if self.zoom_timer:
            self.after_cancel(self.zoom_timer)
        self.zoom_timer = self.after(300, lambda: self._render_image_on_canvas(use_fast_quality=False))

    def on_canvas_double_click(self, event: tk.Event):
        if self.current_display_path:
            try:
                if sys.platform == 'win32':
                    os.startfile(self.current_display_path)
                elif sys.platform == 'darwin':
                    subprocess.run(['open', self.current_display_path])
                else:
                    subprocess.run(['xdg-open', self.current_display_path])
            except Exception as e:
                messagebox.showerror('Error', f'Could not open file: {e}')

    def on_canvas_right_click(self, event: tk.Event):
        if not self.current_display_path:
            return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label='Copy Path', command=lambda: self.clipboard_append(self.current_display_path))
        menu.add_command(label='Open Containing Folder', command=self.open_containing_folder)
        menu.add_separator()
        menu.add_command(label='Search for Similar', command=self.search_for_similar)
        menu.tk_popup(event.x_root, event.y_root)

    def open_containing_folder(self):
        if self.current_display_path:
            folder = os.path.dirname(self.current_display_path)
            if sys.platform == 'win32':
                subprocess.run(['explorer', '/select,', self.current_display_path])
            elif sys.platform == 'darwin':
                subprocess.run(['open', '-R', self.current_display_path])
            else:
                subprocess.run(['xdg-open', folder])

    def search_for_similar(self):
        if self.current_display_path:
            self.query_image_path = self.current_display_path
            self.query_image_var.set(os.path.basename(self.current_display_path))
            self.query_text_var.set('')
            self.threaded_search()

def cli_mode():
    parser = argparse.ArgumentParser(description='SigLIP 2 NaFlex Image/Video Search CLI')
    parser.add_argument('--index', type=str, help='Path to the folder to index')
    parser.add_argument('--search-text', type=str, help='Text to search for')
    parser.add_argument('--search-image', type=str, help='Image to search with')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results to return')
    parser.add_argument('--db', type=str, default='siglip2_embeddings.db', help='Database file path')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'dml'], help='Force a specific device')
    parser.add_argument('--max-patches', type=int, default=256, help='Max patches for the model')
    parser.add_argument('--cleanup', action='store_true', help='Clean up orphaned embeddings from the database')
    args = parser.parse_args()
    device_str, msg = setup_device(args.device)
    print(f'Device: {msg}')
    device = torch.device('cuda' if device_str == 'cuda' else 'cpu')
    if device_str == 'dml' and torch_directml:
        device = torch_directml.device()
    dtype = torch.float16 if device_str == 'cuda' else torch.float32
    init_db(args.db)
    if args.cleanup:
        print('Cleaning up orphaned embeddings...')
        count = cleanup_orphaned_entries(args.db)
        print(f'Removed {count} orphaned embeddings.')
    if not (args.index or args.search_text or args.search_image):
        print('No action specified. Use --index, --search-text, or --search-image.')
        return
    print(f'Loading model {DEFAULT_MODEL}...')
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL)
    attn_implementation = 'sdpa' if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else 'eager'
    model = Siglip2Model.from_pretrained(DEFAULT_MODEL, torch_dtype=dtype, attn_implementation=attn_implementation).to(device)
    model.eval()
    print('Model loaded.')
    if args.index:
        print(f'\nIndexing folder: {args.index}')
        index_files(args.index, device, processor, model, args.db, max_num_patches=args.max_patches)
        print('Indexing complete.')
    if args.search_text or args.search_image:
        print('\nPerforming search...')
        query_embedding = get_query_embedding(args.search_text, args.search_image, device, processor, model, args.max_patches)
        if query_embedding is None:
            print('Error: Could not generate query embedding.')
            return
        results = search_db(query_embedding, args.db, args.top_k)
        if not results:
            print('\nNo results found.')
        else:
            print(f'\n--- Top {len(results)} Results ---')
            for i, (path, score, ftype) in enumerate(results, 1):
                print(f'{i:2d}. [{ftype.upper()}] Score: {score:.4f} | {os.path.basename(path)}')
            print('-' * 20)
if __name__ == '__main__':
    if len(sys.argv) > 1:
        cli_mode()
    else:
        print('Starting SigLIP 2 NaFlex GUI...')
        app = SigLIP2NaFlexApp()
        app.mainloop()