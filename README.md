# ClipSort - Intelligent Image Clustering

ClipSort is a web application that leverages OpenAI's [CLIP](https://github.com/openai/CLIP) model to intelligently organize and cluster images based on their visual content. By uploading a batch of mixed images, ClipSort analyzes them using deep learning embeddings and groups semantically similar images together using K-Means clustering.

## Features

*   **Batch Image Upload**: Upload multiple images at once (JPG, PNG, BMP, GIF, WEBP).
*   **AI-Powered Clustering**: Uses the CLIP model (ViT-B/32) to understand image content and K-Means to group them.
*   **Customizable Clusters**: Choose the number of clusters (groups) you want to generate (2-10).
*   **Visual Preview**: View uploaded images and the resulting clusters directly in the browser.
*   **Download Results**: Download the organized clusters as a ZIP file.
*   **Persistent Tracking**: Uses a local SQLite database to track uploaded images and their assigned clusters (Local mode).

## Tech Stack

*   **Backend Framework**: Flask (Python)
*   **Machine Learning**: PyTorch, Transformers (Hugging Face), Scikit-learn
*   **Model**: OpenAI CLIP (`openai/clip-vit-base-patch32`)
*   **Database**: SQLAlchemy (SQLite)
*   **Frontend**: HTML5, CSS3 (Jinja2 Templates)
*   **Deployment**: Vercel (Serverless)

## Installation & Local Development

To run ClipSort locally, follow these steps:

### Prerequisites

*   Python 3.8 or higher
*   `pip` (Python package manager)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ImgSort
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```

5.  **Access the app:**
    Open your browser and navigate to `http://127.0.0.1:5000`.

## Usage

1.  **Upload Images**: Click "Choose Files" and select the images you want to sort. Click "Upload Images".
2.  **Cluster**: Enter the desired number of clusters (e.g., 4) in the input box and click "Cluster Images".
    *   *Note: This process may take a moment depending on the number of images and your hardware, as it runs the CLIP model.*
3.  **Review**: Scroll down to see your images grouped into "Cluster A", "Cluster B", etc.
4.  **Download**: Click "Download Clustered Images (ZIP)" to get a compressed file with your sorted folders.
5.  **Reset**: Use the "Clear All Cache" button to remove all images and start over.

## Project Structure

```
/
├── app.py               # Main Flask application
├── api/
│   └── index.py         # Entry point for Vercel serverless function
├── templates/
│   └── index.html       # Frontend HTML template
├── clip_images.db       # SQLite database (generated)
├── requirements.txt     # Python dependencies
├── vercel.json          # Vercel configuration
└── README.md            # Project documentation
```

## License

[MIT License](LICENSE)