import os
import shutil
import io
import zipfile
from collections import defaultdict
from flask import Flask, request, redirect, url_for, render_template, send_file, flash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = "uploads"
CLUSTERED_FOLDER = "clusters"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLUSTERED_FOLDER'] = CLUSTERED_FOLDER

# Load CLIP model and processor globally
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print('DEBUG processor type:', type(processor))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Ensure the upload and cluster folders exist and clear them on startup
for folder in [UPLOAD_FOLDER, CLUSTERED_FOLDER]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# SQLAlchemy setup
Base = declarative_base()
DB_PATH = 'sqlite:///clip_images.db'
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class ImageModel(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow)
    cluster = Column(Integer, nullable=True)

Base.metadata.create_all(bind=engine)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    MAX_PREVIEW = 20  # Limit for previewed images
    uploaded_images = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)][:MAX_PREVIEW]
    clustered_folders = [f for f in os.listdir(CLUSTERED_FOLDER) if os.path.isdir(os.path.join(CLUSTERED_FOLDER, f))]
    clustered_images = {}
    cluster_names = {}
    for idx, folder in enumerate(clustered_folders):
        folder_path = os.path.join(CLUSTERED_FOLDER, folder)
        images = [f for f in os.listdir(folder_path) if allowed_file(f)][:MAX_PREVIEW]
        # Use letters for cluster names: Cluster A, Cluster B, ...
        cluster_name = f"Cluster {chr(65+idx)}"
        cluster_names[folder] = cluster_name
        clustered_images[folder] = images
    return render_template('index.html', uploaded_images=uploaded_images, clustered_folders=clustered_folders, clustered_images=clustered_images, cluster_names=cluster_names)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))
    files = request.files.getlist('file')
    uploaded = 0
    db = SessionLocal()
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Add to DB if not already present
            if not db.query(ImageModel).filter_by(filename=filename).first():
                db.add(ImageModel(filename=filename))
                db.commit()
            uploaded += 1
    db.close()
    if uploaded:
        flash(f'{uploaded} file(s) uploaded successfully!')
    else:
        flash('No valid images uploaded!')
    return redirect(url_for('home'))

@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings():
    images = []
    filenames = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if allowed_file(filename):
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            filenames.append(filename)
    if not images:
        flash('No images to cluster!')
        return redirect(url_for('home'))
    # Batching for embeddings
    BATCH_SIZE = 32  # Adjust as needed for your hardware
    all_embeddings = []
    for i in range(0, len(images), BATCH_SIZE):
        batch_images = images[i:i+BATCH_SIZE]
        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        all_embeddings.append(embeddings.cpu().numpy())
    np_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save("image_embeddings.npy", np_embeddings)
    with open("image_filenames.txt", "w") as f:
        for fn in filenames:
            f.write(fn + "\n")
    num_clusters = int(request.form.get('n_clusters', 4))
    if os.path.exists(CLUSTERED_FOLDER):
        shutil.rmtree(CLUSTERED_FOLDER)
    os.makedirs(CLUSTERED_FOLDER, exist_ok=True)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(np_embeddings)
    cluster_folders = []
    for i in range(num_clusters):
        cluster_path = os.path.join(CLUSTERED_FOLDER, f"cluster_{i+1}")
        os.makedirs(cluster_path, exist_ok=True)
        cluster_folders.append(cluster_path)
    db = SessionLocal()
    for filename, label in zip(filenames, labels):
        src_path = os.path.join(UPLOAD_FOLDER, filename)
        dst_path = os.path.join(cluster_folders[label], filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        # Update cluster assignment in DB
        img = db.query(ImageModel).filter_by(filename=filename).first()
        if img:
            img.cluster = label + 1  # cluster_1, cluster_2, ...
            db.commit()
    db.close()
    flash(f'Images clustered into {num_clusters} folders!')
    for cache_file in ["image_embeddings.npy", "image_filenames.txt"]:
        if os.path.exists(cache_file):
            os.remove(cache_file)
    return redirect(url_for('home'))

@app.route('/download_clusters')
def download_clusters():
    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(CLUSTERED_FOLDER):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, start=CLUSTERED_FOLDER)
                zipf.write(filepath, arcname)
    zip_stream.seek(0)
    return send_file(zip_stream, download_name="clusters.zip", as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/clusters/<cluster>/<filename>')
def clustered_file(cluster, filename):
    return send_file(os.path.join(app.config['CLUSTERED_FOLDER'], cluster, filename))

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    for folder in [UPLOAD_FOLDER, CLUSTERED_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    flash('All uploaded and clustered images have been cleared!')
    return redirect(url_for('home'))

if __name__ == '__main__':
    print('Starting Flask server on http://127.0.0.1:5000 ...')
    app.run(debug=True, host='0.0.0.0', port=5000) 