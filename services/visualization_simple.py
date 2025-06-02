"""
Visualization module for image embeddings using simple HTML and SVG.
Generate real embedding using the configured model
from image_understanding import embedding_model
embedding_vector = embedding_model.encode(image_url, convert_to_tensor=True).tolist()
original_filename": "image.jpg",
"""
import os
import uuid
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

# Import the vector database module
try:
    from db.vector_db import get_all_embeddings
except ImportError:
    # Mock function if vector_db is not available
    def get_all_embeddings(*args, **kwargs):
        return {"embeddings": [[]], "metadata": [[]], "ids": [[]]}

# Configure logging
logger = logging.getLogger(__name__)

async def generate_embedding_visualization(
    team_id: Optional[uuid.UUID] = None, 
    method: str = "pca",
    include_images: bool = False
) -> Dict[str, Any]:
    """Generate visualization for image embeddings"""
    try:
        # Get all embeddings from ChromaDB
        results = get_all_embeddings(team_id=team_id, limit=1000)
        
        if not results["embeddings"] or len(results["embeddings"][0]) == 0:
            return {
                "success": False,
                "error": "No embeddings found for visualization"
            }
        
        # Convert embeddings to numpy array
        embeddings = np.array(results["embeddings"][0])
        metadata = results["metadata"][0]
        ids = results["ids"][0]
        
        # Apply dimensionality reduction
        if method.lower() == "tsne":
            try:
                # Try to import TSNE
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, 
                               perplexity=min(30, len(embeddings)-1) if len(embeddings) > 30 else 3)
                reduced_data = reducer.fit_transform(embeddings)
                title = "t-SNE Visualization of Image Embeddings"
            except ImportError:
                # Fall back to simple table if TSNE is not available
                return create_text_visualization(embeddings, metadata, ids, "t-SNE not available")
        else:
            try:
                # Try to import PCA
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
                reduced_data = reducer.fit_transform(embeddings)
                title = "PCA Visualization of Image Embeddings"
            except ImportError:
                # Fall back to simple table if PCA is not available
                return create_text_visualization(embeddings, metadata, ids, "PCA not available")
        
        # Create simple HTML visualization
        html_content = create_simple_html_visualization(reduced_data, metadata, ids, title)
        
        return {
            "success": True,
            "visualization_html": html_content,
            "method": method,
            "embedding_count": len(embeddings)
        }
    
    except Exception as e:
        logger.error(f"Error generating embedding visualization: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def create_simple_html_visualization(
    reduced_data: np.ndarray, 
    metadata_list: List[Dict[str, Any]],
    ids: List[str],
    title: str
) -> str:
    """Create a simple HTML visualization without using matplotlib or plotly"""
    # Generate HTML content
    html_parts = [f"<h2>{title}</h2>"]
    html_parts.append("<div style='overflow-x: auto;'>")
    html_parts.append("<table border='1' cellspacing='0' cellpadding='5'>")
    html_parts.append("<tr><th>ID</th><th>X</th><th>Y</th><th>Metadata</th></tr>")
    
    # Add rows for each data point
    for i, (x, y) in enumerate(reduced_data):
        if i >= len(ids) or i >= len(metadata_list):
            break
            
        doc_id = ids[i]
        metadata = metadata_list[i]
        
        # Format metadata as string
        metadata_str = "<br>".join([f"{k}: {v}" for k, v in metadata.items() if k and v])
        
        # Add table row
        html_parts.append("<tr>")
        html_parts.append(f"<td>{doc_id}</td>")
        html_parts.append(f"<td>{x:.4f}</td>")
        html_parts.append(f"<td>{y:.4f}</td>")
        html_parts.append(f"<td>{metadata_str}</td>")
        html_parts.append("</tr>")
    
    html_parts.append("</table>")
    html_parts.append("</div>")
    
    # Add SVG visualization
    html_parts.append(create_svg_scatter(reduced_data, ids))
    
    return "".join(html_parts)

def create_svg_scatter(reduced_data: np.ndarray, ids: List[str]) -> str:
    """Create a simple SVG scatter plot"""
    if len(reduced_data) == 0:
        return "<p>No data to visualize</p>"
    
    # Calculate bounds
    min_x = min(reduced_data[:, 0])
    max_x = max(reduced_data[:, 0])
    min_y = min(reduced_data[:, 1])
    max_y = max(reduced_data[:, 1])
    
    # Add margins
    margin = 0.1
    width = max_x - min_x
    height = max_y - min_y
    min_x -= width * margin
    max_x += width * margin
    min_y -= height * margin
    max_y += height * margin
    
    # SVG dimensions
    svg_width = 800
    svg_height = 500
    
    # Function to convert data coordinates to SVG coordinates
    def to_svg_x(x):
        return ((x - min_x) / (max_x - min_x)) * svg_width
    
    def to_svg_y(y):
        return svg_height - ((y - min_y) / (max_y - min_y)) * svg_height
    
    # Create SVG header
    svg = [
        f'<svg width="{svg_width}" height="{svg_height}" '
        f'viewBox="0 0 {svg_width} {svg_height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    
    # Add points
    for i, (x, y) in enumerate(reduced_data):
        svg_x = to_svg_x(x)
        svg_y = to_svg_y(y)
        point_id = ids[i] if i < len(ids) else f"point_{i}"
        
        svg.append(
            f'<circle cx="{svg_x}" cy="{svg_y}" r="4" fill="#3366cc" '
            f'stroke="#ffffff" stroke-width="1" opacity="0.7">'
            f'<title>{point_id}</title></circle>'
        )
    
    # Add axes
    origin_x = to_svg_x(0)
    origin_y = to_svg_y(0)
    
    # X-axis
    svg.append(
        f'<line x1="0" y1="{origin_y}" x2="{svg_width}" y2="{origin_y}" '
        f'stroke="#999999" stroke-width="1" stroke-dasharray="5,5" />'
    )
    
    # Y-axis
    svg.append(
        f'<line x1="{origin_x}" y1="0" x2="{origin_x}" y2="{svg_height}" '
        f'stroke="#999999" stroke-width="1" stroke-dasharray="5,5" />'
    )
    
    # Close SVG
    svg.append('</svg>')
    
    return "".join(svg)

def create_text_visualization(
    embeddings: np.ndarray, 
    metadata_list: List[Dict[str, Any]],
    ids: List[str],
    message: str
) -> Dict[str, Any]:
    """Create a text-based visualization when dimensionality reduction is not available"""
    html_parts = [f"<h2>{message}</h2>"]
    html_parts.append("<p>Showing first 10 embeddings:</p>")
    html_parts.append("<div style='overflow-x: auto;'>")
    html_parts.append("<table border='1' cellspacing='0' cellpadding='5'>")
    html_parts.append("<tr><th>ID</th><th>Metadata</th><th>First 5 embedding values</th></tr>")
    
    # Add rows for first 10 embeddings
    for i, embedding in enumerate(embeddings[:10]):
        if i >= len(ids) or i >= len(metadata_list):
            break
            
        doc_id = ids[i]
        metadata = metadata_list[i]
        
        # Format metadata as string
        metadata_str = "<br>".join([f"{k}: {v}" for k, v in metadata.items() if k and v])
        
        # Format first 5 embedding values
        embedding_str = ", ".join([f"{v:.4f}" for v in embedding[:5]])
        
        # Add table row
        html_parts.append("<tr>")
        html_parts.append(f"<td>{doc_id}</td>")
        html_parts.append(f"<td>{metadata_str}</td>")
        html_parts.append(f"<td>[{embedding_str}, ...]</td>")
        html_parts.append("</tr>")
    
    html_parts.append("</table>")
    html_parts.append("</div>")
    
    return {
        "success": True,
        "visualization_html": "".join(html_parts),
        "method": "table",
        "embedding_count": len(embeddings)
    }

