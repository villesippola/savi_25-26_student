import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import ModelBetterCNN # Import your model class from Task 1

def sliding_window(image, step_size, window_size):
    """Generator that yields (x, y, window)"""
    h, w = image.shape
    for y in range(0, h - window_size + 1, step_size):
        for x in range(0, w - window_size + 1, step_size):
            yield (x, y, image[y:y + window_size, x:x + window_size])

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size = 28  # MNIST standard
    step_size = 5     # Stride
    threshold = 0.95  # Confidence threshold

    # 2. Load Model
    model = ModelBetterCNN() # Initialize your architecture
    # Load weights (adjust path to your best.pkl)
    checkpoint = torch.load("best.pkl", map_location=device) 
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 3. Load Test Image (from your Task 2 dataset)
    # Use 0 (grayscale) because model expects 1 channel
    image_path = "data/test/images/0.png" 
    original_img = cv2.imread(image_path, 0) 
    
    # 4. Preprocessing Transform (Must match training!)
    # Usually ToTensor() scales 0-255 to 0-1. 
    # If you used specific Mean/Std normalization in Task 1, add it here.
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)) # Uncomment if used in training
    ])

    # List to store detections: [x, y, x2, y2, score, label]
    detections = []

    # 5. Run Sliding Window
    print("Running sliding window...")
    for (x, y, window) in sliding_window(original_img, step_size, window_size):
        
        # Prepare input
        input_tensor = transform(window).unsqueeze(0).to(device) # Add batch dim [1, 1, 28, 28]

        with torch.no_grad():
            output = model(input_tensor)
            
            # If your model outputs raw logits, apply Softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Get max probability and class
            score, predicted_class = torch.max(probabilities, 1)
            score = score.item()
            
            # Filter by probability
            if score > threshold:
                # Save detection (x, y, w, h, score, label)
                print(f"Found digit {predicted_class.item()} at ({x}, {y}) with conf {score:.2f}")
                detections.append([x, y, x + window_size, y + window_size, score, predicted_class.item()])

    # 6. Visualization
    output_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2, score, label) in detections:
        # Draw rectangle
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # Put label text
        cv2.putText(output_img, f"{label}:{score:.2f}", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imwrite("sliding_window_result.png", output_img)
    plt.imshow(output_img)
    plt.show()

if __name__ == "__main__":
    main()