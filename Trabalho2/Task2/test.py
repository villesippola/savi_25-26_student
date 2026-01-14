import matplotlib.pyplot as plt
import argparse
import pathlib
import numpy as np
import typing
from collections import Counter

colors = ["blue", "green", "cyan", "red", "yellow", "magenta", "peru", "azure", "slateblue", "plum"]

def plot_bbox(ax, bbox_XYXY, label):
    xmin, ymin, xmax, ymax = bbox_XYXY
    ax.plot(
        [xmin, xmin, xmax, xmax, xmin],
        [ymin, ymax, ymax, ymin, ymin],
        color=colors[label], 
        label=str(label))

def read_labels(label_path: pathlib.Path) -> typing.Tuple[np.ndarray]:
    if not label_path.is_file():
        return np.array([]), np.array([])
    labels = []
    BBOXES_XYXY = []
    with open(label_path, "r") as fp:
        lines = fp.readlines()
        # Skip header if present
        start_idx = 1 if len(lines) > 0 and "label" in lines[0] else 0
        for line in lines[start_idx:]:
            label, xmin, ymin, xmax, ymax = [int(_) for _ in line.split(",")]
            labels.append(label)
            BBOXES_XYXY.append([xmin, ymin, xmax, ymax])
    return np.array(labels), np.array(BBOXES_XYXY)

def analyze_dataset_stats(label_dir: pathlib.Path):
    """Calculates and plots statistics for the dataset."""
    all_labels = []
    digits_per_image = []
    widths = []
    heights = []

    label_files = list(label_dir.glob("*.txt"))
    print(f"Analyzing {len(label_files)} label files for statistics...")

    for label_file in label_files:
        labels, bboxes = read_labels(label_file)
        
        # 1. Collect Class Counts
        if len(labels) > 0:
            all_labels.extend(labels)
        
        # 2. Collect Digits per image count
        digits_per_image.append(len(labels))
        
        # 3. Collect Sizes (Width/Height)
        if len(bboxes) > 0:
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox
                w = xmax - xmin
                h = ymax - ymin
                widths.append(w)
                heights.append(h)

    # --- Plotting ---
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten() # Flatten to 1D array for easier indexing [0, 1, 2, 3]
    
    # Plot 1: Histogram of Digits per Image
    if digits_per_image:
        bins = np.arange(min(digits_per_image), max(digits_per_image) + 2) - 0.5
        axes[0].hist(digits_per_image, bins=bins, rwidth=0.8, color='skyblue', edgecolor='black')
        axes[0].set_title("Histogram: Digits per Image")
        axes[0].set_xlabel("Number of Digits")
        axes[0].set_ylabel("Frequency")
        axes[0].set_xticks(range(min(digits_per_image), max(digits_per_image) + 1))
        axes[0].grid(axis='y', alpha=0.5)

    # Plot 2: Class Distribution
    if all_labels:
        class_counts = Counter(all_labels)
        for i in range(10): # Ensure 0-9 exist
            if i not in class_counts: class_counts[i] = 0
        
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        
        axes[1].bar(classes, counts, color='lightgreen', edgecolor='black')
        axes[1].set_title("Class Distribution")
        axes[1].set_xlabel("Digit Class (0-9)")
        axes[1].set_ylabel("Count")
        axes[1].set_xticks(classes)
        axes[1].grid(axis='y', alpha=0.5)

    # Plot 3: Width Distribution
    if widths:
        axes[2].hist(widths, bins=20, color='salmon', edgecolor='black')
        axes[2].set_title("Digit Width Distribution")
        axes[2].set_xlabel("Width (pixels)")
        axes[2].set_ylabel("Frequency")
        axes[2].grid(axis='y', alpha=0.5)

    # Plot 4: Height Distribution
    if heights:
        axes[3].hist(heights, bins=20, color='orchid', edgecolor='black')
        axes[3].set_title("Digit Height Distribution")
        axes[3].set_xlabel("Height (pixels)")
        axes[3].set_ylabel("Frequency")
        axes[3].grid(axis='y', alpha=0.5)

    plt.tight_layout()
    plt.savefig("dataset_statistics.png")
    print("Graphs saved to 'dataset_statistics.png'.")

    # --- Print Text Stats ---
    print("\n" + "="*30)
    print("       DATASET STATISTICS       ")
    print("="*30)
    print(f"Total Images: {len(label_files)}")
    print(f"Total Digits: {len(all_labels)}")
    if widths:
        print(f"Avg Digit Width:  {np.mean(widths):.2f} px")
        print(f"Avg Digit Height: {np.mean(heights):.2f} px")
        print(f"Min Width: {np.min(widths)} | Max Width: {np.max(widths)}")
        print(f"Min Height: {np.min(heights)} | Max Height: {np.max(heights)}")
    print("="*30 + "\n")

    # Only show if in an environment that supports it
    try:
        plt.show()
    except:
        pass

def visualize_mosaic(image_dir: pathlib.Path, label_dir: pathlib.Path, rows=4, cols=4):
    """Displays a mosaic of random images with bounding boxes."""
    impaths = list(image_dir.glob("*.png"))
    
    if not impaths:
        print("No images found to visualize.")
        return

    # Select random images
    num_images = rows * cols
    selected_impaths = np.random.choice(impaths, size=min(num_images, len(impaths)), replace=False)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()

    print(f"Visualizing mosaic of {len(selected_impaths)} images...")

    for i, impath in enumerate(selected_impaths):
        label_path = label_dir.joinpath(f"{impath.stem}.txt")
        labels, bboxes_XYXY = read_labels(label_path)
        
        im = plt.imread(str(impath))
        
        axes[i].imshow(im, cmap="gray")
        axes[i].set_title(f"Image {impath.stem}")
        axes[i].axis('off')
        
        if len(bboxes_XYXY) > 0:
            for bbox, label in zip(bboxes_XYXY, labels):
                plot_bbox(axes[i], bbox, label)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("dataset_mosaic.png")
    print("Mosaic saved to 'dataset_mosaic.png'.")
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Path to the dataset directory (containing 'images' and 'labels' folders)")
    parser.add_argument("--no-stats", action="store_true", help="Skip statistical analysis")
    parser.add_argument("--no-viz", action="store_true", help="Skip mosaic visualization")
    args = parser.parse_args()

    base_path = pathlib.Path(args.directory)
    image_dir = base_path.joinpath("images")
    label_dir = base_path.joinpath("labels")

    if not image_dir.exists() or not label_dir.exists():
        print(f"Error: Could not find 'images' or 'labels' directories in {base_path}")
        exit(1)

    if not args.no_stats:
        analyze_dataset_stats(label_dir)

    if not args.no_viz:
        visualize_mosaic(image_dir, label_dir)