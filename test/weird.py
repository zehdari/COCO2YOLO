#!/usr/bin/env python3
"""
Simplified COCO to YOLO Converter for Video Datasets
Converts COCO bounding box annotations from zipped annotation files to YOLO format.
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import yaml
import argparse


def coco_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert COCO bounding box [x, y, width, height] to YOLO format [x_center, y_center, width, height].
    All values are normalized to [0, 1].
    """
    x, y, width, height = bbox
    
    # Convert to center coordinates
    x_center = x + width / 2
    y_center = y + height / 2
    
    # Normalize to image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def discover_label_mapping(zip_path: Path, temp_dir: Path) -> Dict[str, int]:
    """
    Extract COCO annotations from zip file and discover all categories.
    Returns a mapping of category names to YOLO class IDs.
    """
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find the instances_default.json file
    json_file = None
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file == "instances_default.json":
                json_file = Path(root) / file
                break
        if json_file:
            break
    
    if not json_file or not json_file.exists():
        raise FileNotFoundError(f"No instances_default.json found in {zip_path}")
    
    # Load COCO data
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create label mapping from categories, sorted alphabetically for consistency
    categories = sorted([cat['name'] for cat in coco_data['categories']])
    label_mapping = {name: idx for idx, name in enumerate(categories)}
    
    print(f"Discovered {len(categories)} categories: {categories}")
    print(f"Label mapping: {label_mapping}")
    
    return label_mapping


def extract_and_convert_annotations(zip_path: Path, temp_dir: Path, label_mapping: Dict[str, int]) -> Dict[str, str]:
    """
    Extract COCO annotations from zip file and convert to YOLO format.
    Returns a mapping of image filenames to their YOLO annotation content.
    """
    annotations = {}
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find the instances_default.json file
    json_file = None
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file == "instances_default.json":
                json_file = Path(root) / file
                break
        if json_file:
            break
    
    if not json_file or not json_file.exists():
        print(f"Warning: No instances_default.json found in {zip_path}")
        return annotations
    
    # Load COCO data
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    category_mapping = {cat['id']: cat['name'] for cat in coco_data['categories']}
    image_mapping = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Convert each image's annotations
    for image_id, image_info in image_mapping.items():
        filename = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']
        
        yolo_lines = []
        
        if image_id in image_annotations:
            for ann in image_annotations[image_id]:
                category_name = category_mapping[ann['category_id']]
                
                # Skip if category not in label mapping
                if category_name not in label_mapping:
                    print(f"Warning: Category '{category_name}' not found in label mapping")
                    continue
                
                class_id = label_mapping[category_name]
                bbox = ann['bbox']  # COCO format: [x, y, width, height]
                
                # Convert to YOLO format
                x_center, y_center, width, height = coco_bbox_to_yolo(bbox, img_width, img_height)
                
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)
        
        # Store annotation content (empty string if no annotations)
        annotations[filename] = '\n'.join(yolo_lines)
    
    return annotations


def create_yolo_yaml(dataset_base: Path, label_mapping: Dict[str, int]):
    """Create YOLO data.yaml file with video subfolder structure."""
    yaml_path = dataset_base / "data.yaml"
    
    # Get class names in order of their IDs
    class_names = [''] * len(label_mapping)
    for name, class_id in label_mapping.items():
        class_names[class_id] = name
    
    # Find all video subfolders in train and val
    train_paths = []
    val_paths = []
    test_paths = []
    
    train_dir = dataset_base / 'images' / 'train'
    val_dir = dataset_base / 'images' / 'val'
    test_dir = dataset_base / 'images' / 'test'
    
    if train_dir.exists():
        for video_folder in sorted(train_dir.iterdir()):
            if video_folder.is_dir():
                train_paths.append(str(video_folder.resolve()))
    
    if val_dir.exists():
        for video_folder in sorted(val_dir.iterdir()):
            if video_folder.is_dir():
                val_paths.append(str(video_folder.resolve()))
    
    if test_dir.exists():
        for video_folder in sorted(test_dir.iterdir()):
            if video_folder.is_dir():
                test_paths.append(str(video_folder.resolve()))
    
    data = {
        'nc': len(label_mapping),
        'names': class_names
    }
    
    # Add paths in YOLO format
    if train_paths:
        data['train'] = train_paths
    if val_paths:
        data['val'] = val_paths
    if test_paths:
        data['test'] = test_paths
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created YOLO data.yaml at {yaml_path}")
    print(f"  Train videos: {len(train_paths)}")
    print(f"  Val videos: {len(val_paths)}")
    if test_paths:
        print(f"  Test videos: {len(test_paths)}")


def process_video_dataset(
    input_dir: Path,
    output_dir: Path,
    val_split: float = 0.2,
    test_split: float = 0.0,
    random_seed: int = 42
):
    """
    Process video dataset with zipped COCO annotations.
    
    Args:
        input_dir: Directory containing videos and annotations folder
        output_dir: Output directory for YOLO dataset
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducible splits
    """
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    annotations_dir = input_dir / "annotations"
    
    # Validate input structure
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    
    # Find video files and corresponding annotation zips
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    video_files = []
    
    for file in input_dir.iterdir():
        if file.suffix.lower() in video_extensions:
            video_files.append(file.stem)  # filename without extension
    
    if not video_files:
        raise ValueError(f"No video files found in {input_dir}")
    
    print(f"Found {len(video_files)} video files")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "__temp__"
    temp_dir.mkdir(exist_ok=True)
    
    # Discover label mapping from first available annotation file
    label_mapping = None
    for video_name in video_files:
        zip_path = annotations_dir / f"{video_name}.zip"
        if zip_path.exists():
            print(f"Discovering categories from {video_name}...")
            discovery_dir = temp_dir / "discovery"
            discovery_dir.mkdir(exist_ok=True)
            try:
                label_mapping = discover_label_mapping(zip_path, discovery_dir)
                break
            except Exception as e:
                print(f"Error discovering categories from {video_name}: {e}")
                continue
            finally:
                if discovery_dir.exists():
                    shutil.rmtree(discovery_dir)
    
    if label_mapping is None:
        raise ValueError("Could not discover categories from any annotation files")
    
    all_annotations = {}
    
    try:
        # Process each video's annotations
        for video_name in video_files:
            zip_path = annotations_dir / f"{video_name}.zip"
            
            if not zip_path.exists():
                print(f"Warning: No annotation zip found for video {video_name}")
                continue
            
            print(f"Processing annotations for {video_name}...")
            
            # Create temporary extraction directory
            extract_dir = temp_dir / video_name
            extract_dir.mkdir(exist_ok=True)
            
            try:
                # Extract and convert annotations
                video_annotations = extract_and_convert_annotations(zip_path, extract_dir, label_mapping)
                
                # Add video prefix to avoid filename conflicts
                for filename, content in video_annotations.items():
                    prefixed_name = f"{video_name}_{filename}"
                    all_annotations[prefixed_name] = content
                
                print(f"Processed {len(video_annotations)} frames from {video_name}")
                
            except Exception as e:
                print(f"Error processing {video_name}: {e}")
                continue
            
            finally:
                # Clean up extraction directory
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
        
        if not all_annotations:
            raise ValueError("No annotations were successfully processed")
        
        print(f"Total processed frames: {len(all_annotations)}")
        
        # Group files by video name for splitting
        video_groups = {}
        for filename in all_annotations.keys():
            # Extract video name from prefixed filename (e.g., "video1_frame_001.jpg" -> "video1")
            video_name = filename.split('_', 1)[0]
            if video_name not in video_groups:
                video_groups[video_name] = []
            video_groups[video_name].append(filename)
        
        # Split videos (not individual frames) into train/val/test
        video_names = list(video_groups.keys())
        
        if test_split > 0:
            train_val_videos, test_videos = train_test_split(
                video_names, test_size=test_split, random_state=random_seed
            )
            train_videos, val_videos = train_test_split(
                train_val_videos, test_size=val_split/(1-test_split), random_state=random_seed
            )
        else:
            train_videos, val_videos = train_test_split(
                video_names, test_size=val_split, random_state=random_seed
            )
            test_videos = []
        
        # Count total frames for logging
        train_frame_count = sum(len(video_groups[video]) for video in train_videos)
        val_frame_count = sum(len(video_groups[video]) for video in val_videos)
        test_frame_count = sum(len(video_groups[video]) for video in test_videos)
        
        print(f"Video split: {len(train_videos)} train videos ({train_frame_count} frames), "
              f"{len(val_videos)} val videos ({val_frame_count} frames), "
              f"{len(test_videos)} test videos ({test_frame_count} frames)")
        
        # Create dataset directories with video subfolders
        splits = [('train', train_videos), ('val', val_videos)]
        if test_videos:
            splits.append(('test', test_videos))
        
        for split_name, split_videos in splits:
            for video_name in split_videos:
                # Create video-specific directories
                video_images_dir = output_dir / 'images' / split_name / video_name
                video_labels_dir = output_dir / 'labels' / split_name / video_name
                video_images_dir.mkdir(parents=True, exist_ok=True)
                video_labels_dir.mkdir(parents=True, exist_ok=True)
                
                # Process all frames for this video
                for filename in video_groups[video_name]:
                    # Remove video prefix to get original frame filename
                    original_filename = filename.split('_', 1)[1]
                    
                    # Create label file
                    label_filename = Path(original_filename).with_suffix('.txt').name
                    label_path = video_labels_dir / label_filename
                    
                    with open(label_path, 'w') as f:
                        f.write(all_annotations[filename])
                    
                    # Create placeholder image file (since we don't have actual images)
                    # You'll need to extract frames from videos separately
                    image_filename = original_filename
                    if not any(original_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        image_filename += '.jpg'
                    
                    placeholder_path = video_images_dir / image_filename
                    placeholder_path.touch()  # Create empty file as placeholder
        
        # Create YOLO yaml file
        create_yolo_yaml(output_dir, label_mapping)
        
        print(f"Dataset created successfully at {output_dir}")
        print("Note: Placeholder image files were created. You'll need to extract actual frames from your videos.")
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description='Convert video COCO annotations to YOLO format')
    parser.add_argument('input_dir', help='Input directory containing videos and annotations')
    parser.add_argument('output_dir', help='Output directory for YOLO dataset')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--test-split', type=float, default=0.0, help='Test split ratio (default: 0.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    try:
        process_video_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            val_split=args.val_split,
            test_split=args.test_split,
            random_seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())