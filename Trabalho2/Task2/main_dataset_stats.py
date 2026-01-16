import matplotlib.pyplot as plt
import argparse
import pathlib
import numpy as np
import typing
from collections import Counter
import json

colors = ["blue", "green", "cyan", "red", "yellow", "magenta", "peru", "azure", "slateblue", "plum"]


def plot_bbox(bbox_XYXY, label, ax):
    """Plot bounding box on given axis"""
    xmin, ymin, xmax, ymax = bbox_XYXY
    ax.plot(
        [xmin, xmin, xmax, xmax, xmin],
        [ymin, ymax, ymax, ymin, ymin],
        color=colors[label], 
        linewidth=2,
        label=str(label))
    
    # Add label text at top-left corner of bbox
    ax.text(xmin, ymin-2, str(label), 
            color=colors[label], 
            fontsize=10, 
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


def read_labels(label_path: pathlib.Path) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Read labels and bounding boxes from file"""
    assert label_path.is_file()
    labels = []
    BBOXES_XYXY = []
    with open(label_path, "r") as fp:
        for line in list(fp.readlines())[1:]:  # Skip header
            label, xmin, ymin, xmax, ymax = [int(_) for _ in line.split(",")]
            labels.append(label)
            BBOXES_XYXY.append([xmin, ymin, xmax, ymax])
    return np.array(labels), np.array(BBOXES_XYXY)


def compute_statistics(base_path: pathlib.Path):
    """
    Compute comprehensive statistics about the dataset
    
    Returns:
        dict: Dictionary containing all statistics
    """
    image_dir = base_path.joinpath("images")
    label_dir = base_path.joinpath("labels")
    impaths = sorted(list(image_dir.glob("*.png")))
    
    # Initialize counters
    all_labels = []
    digits_per_image = []
    digit_sizes = []
    
    print(f"Computing statistics for {len(impaths)} images...")
    
    # Process all images
    for impath in impaths:
        label_path = label_dir.joinpath(f"{impath.stem}.txt")
        labels, bboxes_XYXY = read_labels(label_path)
        
        # Count digits per image
        digits_per_image.append(len(labels))
        
        # Collect all labels
        all_labels.extend(labels.tolist())
        
        # Compute bbox statistics
        for bbox in bboxes_XYXY:
            xmin, ymin, xmax, ymax = bbox
            size = xmax - xmin
            
            digit_sizes.append(size)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    digits_per_image = np.array(digits_per_image)
    digit_sizes = np.array(digit_sizes)
    
    # Compute statistics
    stats = {
        'dataset_info': {
            'total_images': len(impaths),
            'total_digits': len(all_labels),
            'avg_digits_per_image': float(np.mean(digits_per_image)),
            'std_digits_per_image': float(np.std(digits_per_image)),
            'min_digits_per_image': int(np.min(digits_per_image)),
            'max_digits_per_image': int(np.max(digits_per_image)),
        },
        'class_distribution': {
            int(i): int((all_labels == i).sum()) for i in range(10)
        },
        'digit_size': {
            'avg_size': float(np.mean(digit_sizes)),
            'std_size': float(np.std(digit_sizes)),
            'min_size': int(np.min(digit_sizes)),
            'max_size': int(np.max(digit_sizes)),
        }
    }
    
    # Store raw data for plotting
    stats['_raw_data'] = {
        'all_labels': all_labels,
        'digits_per_image': digits_per_image,
        'digit_sizes': digit_sizes,
    }
    
    return stats


def print_statistics(stats: dict):
    """Print statistics in a formatted way"""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    print("\n📊 Dataset Information:")
    print(f"  Total images: {stats['dataset_info']['total_images']}")
    print(f"  Total digits: {stats['dataset_info']['total_digits']}")
    print(f"  Avg digits/image: {stats['dataset_info']['avg_digits_per_image']:.2f} ± {stats['dataset_info']['std_digits_per_image']:.2f}")
    print(f"  Min digits/image: {stats['dataset_info']['min_digits_per_image']}")
    print(f"  Max digits/image: {stats['dataset_info']['max_digits_per_image']}")
    
    print("\n🔢 Class Distribution:")
    for digit, count in sorted(stats['class_distribution'].items()):
        percentage = (count / stats['dataset_info']['total_digits']) * 100
        bar = "█" * int(percentage / 2)
        print(f"  Digit {digit}: {count:5d} ({percentage:5.2f}%) {bar}")


def plot_statistics(stats: dict, save_path: str = "statistics.png"):
    """
    Create comprehensive visualization of dataset statistics
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Get raw data
    raw = stats['_raw_data']
    
    # Prepare data for consistent use
    digit_labels = list(range(10))
    digit_counts = [stats['class_distribution'][i] for i in digit_labels]
    
    # 1. Class Distribution (Bar Chart)
    ax1 = plt.subplot(2, 4, 1)
    bars = ax1.bar(digit_labels, digit_counts, color=colors[:10], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Digit Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(digit_labels)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    total = sum(digit_counts)
    for bar, count in zip(bars, digit_counts):
        height = bar.get_height()
        if height > 0:  # Only add label if there's data
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/total*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9)
    
    # 2. Digits per Image (Histogram)
    ax2 = plt.subplot(2, 4, 2)
    unique, counts = np.unique(raw['digits_per_image'], return_counts=True)
    ax2.bar(unique, counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Digits', fontsize=12)
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.set_title('Digits per Image Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticks(unique)
    
    # Add count labels
    for x, y in zip(unique, counts):
        ax2.text(x, y, str(y), ha='center', va='bottom', fontsize=10)
    
    # 3. Digit Size Distribution
    ax3 = plt.subplot(2, 4, 3)
    ax3.hist(raw['digit_sizes'], bins=30, color='green', alpha=0.7, edgecolor='black')
    ax3.axvline(stats['digit_size']['avg_size'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {stats['digit_size']['avg_size']:.1f}")
    ax3.set_xlabel('Size (pixels)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Digit Size Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Statistics plot saved to {save_path}")
    plt.show()


def visualize_mosaic(base_path: pathlib.Path, num_images: int = 16, save_path: str = "mosaic.png"):
    """
    Visualize multiple images in a grid mosaic with bounding boxes
    
    Args:
        base_path: Path to dataset directory (train or test)
        num_images: Number of images to display in mosaic (default: 16 for 4x4 grid)
        save_path: Path to save the mosaic image
    """
    image_dir = base_path.joinpath("images")
    label_dir = base_path.joinpath("labels")
    impaths = list(image_dir.glob("*.png"))
    
    # Randomly sample images if we have more than needed
    if len(impaths) > num_images:
        impaths = np.random.choice(impaths, num_images, replace=False)
    else:
        num_images = len(impaths)
    
    # Calculate grid dimensions (try to make it square-ish)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    rows = grid_size
    cols = grid_size
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    
    # Flatten axes array for easier indexing
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each image with its bounding boxes
    for idx, impath in enumerate(impaths):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Read image
        im = plt.imread(str(impath))
        ax.imshow(im, cmap="gray")
        
        # Read and plot bounding boxes
        label_path = label_dir.joinpath(f"{impath.stem}.txt")
        labels, bboxes_XYXY = read_labels(label_path)
        
        for bbox, label in zip(bboxes_XYXY, labels):
            plot_bbox(bbox, label, ax)
        
        # Set title with image name and number of digits
        ax.set_title(f"Image {impath.stem} ({len(labels)} digits)", fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"🖼️  Mosaic saved to {save_path}")
    plt.show()


def visualize_single(base_path: pathlib.Path, image_idx: int = 0, save_path: str = "example_image.png"):
    """
    Visualize a single image with bounding boxes
    
    Args:
        base_path: Path to dataset directory (train or test)
        image_idx: Index of image to display
        save_path: Path to save the image
    """
    image_dir = base_path.joinpath("images")
    label_dir = base_path.joinpath("labels")
    impaths = sorted(list(image_dir.glob("*.png")))
    
    if image_idx >= len(impaths):
        print(f"Image index {image_idx} out of range. Dataset has {len(impaths)} images.")
        return
    
    impath = impaths[image_idx]
    
    # Read image
    im = plt.imread(str(impath))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(im, cmap="gray")
    
    # Read and plot bounding boxes
    label_path = label_dir.joinpath(f"{impath.stem}.txt")
    labels, bboxes_XYXY = read_labels(label_path)
    
    for bbox, label in zip(bboxes_XYXY, labels):
        plot_bbox(bbox, label, ax)
    
    ax.set_title(f"Image {impath.stem} - {len(labels)} digits", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"🖼️  Image saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MNIST detection dataset with statistics")
    parser.add_argument("directory", help="Path to dataset directory (e.g., data/mnist_detection/train/)")
    parser.add_argument("--mode", choices=["mosaic", "single", "stats", "all"], default="all",
                        help="Visualization mode: mosaic, single, stats (statistics only), or all")
    parser.add_argument("--num-images", type=int, default=16,
                        help="Number of images to display in mosaic mode (default: 16)")
    parser.add_argument("--image-idx", type=int, default=0,
                        help="Image index to display in single mode (default: 0)")
    parser.add_argument("--save-mosaic", type=str, default="mosaic.png",
                        help="Path to save mosaic visualization")
    parser.add_argument("--save-single", type=str, default="example_image.png",
                        help="Path to save single image visualization")
    parser.add_argument("--save-stats", type=str, default="statistics.png",
                        help="Path to save statistics plot")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Path to save statistics as JSON (optional)")
    
    args = parser.parse_args()
    base_path = pathlib.Path(args.directory)
    
    # Compute statistics (needed for all modes)
    stats = compute_statistics(base_path)
    
    if args.mode in ["stats", "all"]:
        print_statistics(stats)
        plot_statistics(stats, args.save_stats)
        
        # Save to JSON if requested
        if args.save_json:
            # Remove raw data before saving to JSON
            stats_to_save = {k: v for k, v in stats.items() if k != '_raw_data'}
            with open(args.save_json, 'w') as f:
                json.dump(stats_to_save, f, indent=4)
            print(f"💾 Statistics saved to {args.save_json}")
    
    if args.mode in ["mosaic", "all"]:
        visualize_mosaic(base_path, args.num_images, args.save_mosaic)
    
    if args.mode in ["single", "all"]:
        visualize_single(base_path, args.image_idx, args.save_single)