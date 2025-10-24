import os
import torch
import torch.nn as nn
import timm
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms
import numpy as np
import io

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best_vision_model.pth' # Path to your trained model

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Initialize Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load the Deep Learning Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model onto device: {device}")

# Re-define the model class exactly as it was during training
class FineTuningVisionModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(num_classes=1) # Regression head

    def forward(self, x):
        return self.model(x)

# Load the state dict from the saved file
try:
    image_model = FineTuningVisionModel()
    image_model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # map_location handles CPU/GPU
    image_model.to(device)
    image_model.eval() # Set to evaluation mode
    print("Fine-tuned image model loaded successfully.")
    # Get the necessary image transformations from the loaded model
    data_config = timm.data.resolve_model_data_config(image_model.model)
    image_transforms = timm.data.create_transform(**data_config, is_training=False)
    print("Image transforms configured.")

except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Cannot start server.")
    image_model = None # Flag that model loading failed
except Exception as e:
    print(f"Error loading model: {e}")
    image_model = None


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Backend Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the frontend."""
    if image_model is None:
        return jsonify({'error': 'Model not loaded. Server cannot make predictions.'}), 500

    # Check if an image file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Check if the filename is valid
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        try:
            # Read image file in memory
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # --- Image Preprocessing ---
            img_tensor = image_transforms(img).unsqueeze(0).to(device)

            # --- Prediction ---
            with torch.no_grad():
                log_price_pred = image_model(img_tensor).squeeze().cpu().item()

            # Convert log price back to actual price
            predicted_price = np.expm1(log_price_pred)
            predicted_price = max(0.0, predicted_price) # Ensure non-negative

            # --- Get Description (Optional for future use) ---
            # description = request.form.get('description', '')
            # print(f"Received description: {description}") # Log description

            # Return the prediction
            return jsonify({'predicted_price': f'{predicted_price:.2f}'})

        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Failed to process image or predict price.'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

# --- Run the App ---
if __name__ == '__main__':
    # Make sure the model loaded before starting
    if image_model is not None:
        # Use port 5001 to avoid potential conflicts with other services
        app.run(debug=True, port=5001)
    else:
        print("Model failed to load. Flask server will not start.")
