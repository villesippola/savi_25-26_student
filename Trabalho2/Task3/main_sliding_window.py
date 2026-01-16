#!/usr/bin/env python3
import sys
import os
import cv2
import torch
import glob
import numpy as np
from torchvision import transforms

# -----------------------------------------------------------
# INITIALIZATION & IMPORTS
# -----------------------------------------------------------
# Add the Task1 directory to the system path to allow importing 
# the model class definition without code duplication.
sys.path.append('../Task1')
try:
    from model import ModelBetterCNN
except ImportError:
    print("ERROR: Could not find model.py. Ensure Task1 folder is next to Task3.")
    sys.exit(1)

def main():
    # -------------------------------------------------------
    # 1. CONFIGURATION
    # -------------------------------------------------------
    # Path to the trained model checkpoint (from Task 1)
    model_path = '/home/ville/data/savi_experiments/task1/best.pkl'
    
    # Path to the test images (Dataset Version B is recommended for testing)
    images_path = '/home/ville/data/savi_datasets/versionB_dataset/test/images/*.png' 
    
    # Select the processing device (GPU is highly recommended for sliding window)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------
    # 2. LOAD PRE-TRAINED MODEL
    # -------------------------------------------------------
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    # Initialize the architecture (Must match the one used in Task 1)
    model = ModelBetterCNN().to(device)
    
    # Load weights. Note: 'weights_only=False' is required for older checkpoints 
    # or checkpoints containing numpy data to bypass PyTorch 2.6+ security check.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode (freezes Dropout/BatchNorm)
    model.eval() 
    print("Model loaded successfully.")

    # -------------------------------------------------------
    # 3. PREPARE TRANSFORMS
    # -------------------------------------------------------
    # The input must be converted to a Tensor (0-1 range) to match training conditions
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # -------------------------------------------------------
    # 4. LOAD IMAGES AND RUN SLIDING WINDOW
    # -------------------------------------------------------
    image_files = glob.glob(images_path)
    if len(image_files) == 0:
        print("ERROR: No images found in the specified path.")
        return
    
    # Sort files to ensure consistent order
    image_files.sort()

    print(f"Found {len(image_files)} images. Press any key to next, 'q' to quit.")

    for img_file in image_files:
        print(f"Processing: {img_file}")
        
        # Load image in Grayscale (Mode 0)
        original_image = cv2.imread(img_file, 0)
        
        # Create a color copy to draw the green bounding boxes
        output_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        h, w = original_image.shape
        window_size = 28  # The input size expected by the MNIST model
        step_size = 6     # Stride: Lower = more precise but slower. Higher = faster.

        # --- SLIDING WINDOW LOOP ---
        # Iterate over the image height (y) and width (x)
        for y in range(0, h - window_size + 1, step_size):
            for x in range(0, w - window_size + 1, step_size):
                
                # 1. Crop the window
                window = original_image[y:y+window_size, x:x+window_size]
                
                # 2. Preprocess
                # Convert to Tensor and add batch dimension: [1, 1, 28, 28]
                window_tensor = transform(window).unsqueeze(0).to(device)

                # 3. Inference
                with torch.no_grad():
                    output = model(window_tensor)
                    probabilities = torch.softmax(output, dim=1)
                
                # 4. Get confidence score and predicted class
                score, predicted_class = torch.max(probabilities, 1)
                score = score.item()
                predicted_class = predicted_class.item()

                # 5. Thresholding
                # Since the model was not trained on "background", it will classify 
                # black areas as digits. We use a high threshold to filter noise.
                if score > 0.98: 
                    # Draw Green Rectangle
                    cv2.rectangle(output_image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 1)
                    # Label the digit
                    cv2.putText(output_image, str(predicted_class), (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ---------------------------------------------------------
        # VISUALIZATION & SAVING
        # ---------------------------------------------------------
        # Zoom: Resize image (5x) for better visualization on high-res screens
        scale = 5  
        large_w = w * scale
        large_h = h * scale
        
        # Use INTER_NEAREST to keep the pixel-art look (sharp edges)
        large_image = cv2.resize(output_image, (large_w, large_h), interpolation=cv2.INTER_NEAREST)
        
        # Show result in window
        cv2.imshow("Sliding Window Detection", large_image)
        
        # Auto-save the result to the folder for the report
        filename = os.path.basename(img_file)
        cv2.imwrite(f"resultado_{filename}", large_image)

        # Wait for user input
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
