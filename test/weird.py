#!/usr/bin/env python3
"""
Simplified COCO to YOLO Converter for Video Datasets
Converts COCO bounding box annotations from zipped annotation files to YOLO format.
Extracts actual frames from videos based on COCO annotations.
"""

import os
import json
import shutil
import zipfile
import cv2
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
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


def extract_frame_from_video(video_path: Path, frame_number: int, output_path: Path) -> bool:
    """
    Extract a specific frame from a video file.
    Returns True if successful, False otherwise.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Set frame position (0-indexed)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), frame)
            cap.release()
            return True
        else:
            print(f"Warning: Could not read frame {frame_number} from {video_path}")
            cap.release()
            return False
    except Exception as e:
        print(f"Error extracting frame {frame_number} from {video_path}: {e}")
        return False


def get_frame_number_from_filename(filename: str) -> int:
    """
    Extract frame number from COCO image filename.
    Assumes format like 'frame_001.jpg', '001.jpg', or similar patterns.
    """
    # Try different patterns to extract frame numbers
    patterns = [
        r'frame_(\d+)',     # frame_001.jpg
        r'(\d+)',           # 001.jpg or just numbers
        r'frame(\d+)',      # frame001.jpg
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    
    # If no pattern matches, try to extract any numbers from filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # Use the first number found
        return int(numbers[0])
    
    raise ValueError(f"Could not extract frame number from filename: {filename}")


def extract_and_convert_annotations(zip_path: Path, temp_dir: Path, label_mapping: Dict[str, int], video_path: Path) -> Tuple[Dict[str, str], Set[int]]:
    """
    Extract COCO annotations from zip file and convert to YOLO format.
    Also extract the set of all frame numbers mentioned in the annotations.
    Returns a tuple of (annotations_dict, all_frame_numbers_set).
    """
    annotations = {}
    all_frame_numbers = set()
    
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
        return annotations, all_frame_numbers
    
    # Load COCO data
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    category_mapping = {cat['id']: cat['name'] for cat in coco_data['categories']}
    image_mapping = {img['id']: img for img in coco_data['images']}
    
    # Extract all frame numbers from image filenames
    for image_info in coco_data['images']:
        try:
            frame_num = get_frame_number_from_filename(image_info['file_name'])
            all_frame_numbers.add(frame_num)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
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
        
        try:
            frame_number = get_frame_number_from_filename(filename)
        except ValueError:
            continue
        
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
        annotations[frame_number] = '\n'.join(yolo_lines)
    
    return annotations, all_frame_numbers


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
            video_path = None
            
            # Find the corresponding video file
            for ext in video_extensions:
                potential_video_path = input_dir / f"{video_name}{ext}"
                if potential_video_path.exists():
                    video_path = potential_video_path
                    break
            
            if not video_path:
                print(f"Warning: No video file found for {video_name}")
                continue
                
            if not zip_path.exists():
                print(f"Warning: No annotation zip found for video {video_name}")
                continue
            
            print(f"Processing annotations for {video_name}...")
            
            # Create temporary extraction directory
            extract_dir = temp_dir / video_name
            extract_dir.mkdir(exist_ok=True)
            
            try:
                # Extract and convert annotations
                video_annotations, all_frame_numbers = extract_and_convert_annotations(
                    zip_path, extract_dir, label_mapping, video_path
                )
                
                # Store annotations by frame number instead of filename
                for frame_num, content in video_annotations.items():
                    prefixed_key = f"{video_name}_{frame_num}"
                    all_annotations[prefixed_key] = {
                        'content': content,
                        'frame_number': frame_num,
                        'video_path': video_path
                    }
                
                print(f"Processed {len(video_annotations)} annotated frames from {video_name}")
                print(f"Total frames mentioned in annotations: {len(all_frame_numbers)}")
                
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
        for key, annotation_data in all_annotations.items():
            # Extract video name from prefixed key (e.g., "video1_123" -> "video1")
            video_name = key.split('_', 1)[0]
            if video_name not in video_groups:
                video_groups[video_name] = []
            video_groups[video_name].append(key)
        
        print(f"Total videos found: {len(video_groups)}")
        
        # Split frames within each video between train/val/test
        train_files_by_video = {}
        val_files_by_video = {}
        test_files_by_video = {}
        
        total_train_frames = 0
        total_val_frames = 0
        total_test_frames = 0
        
        for video_name, video_keys in video_groups.items():
            num_frames = len(video_keys)
            print(f"Splitting {num_frames} frames from {video_name}")
            
            if num_frames == 1:
                # Single frame goes to training
                train_files_by_video[video_name] = video_keys
                val_files_by_video[video_name] = []
                test_files_by_video[video_name] = []
                total_train_frames += 1
            elif num_frames == 2:
                # With 2 frames: 1 train, 1 val
                train_files_by_video[video_name] = video_keys[:1]
                val_files_by_video[video_name] = video_keys[1:]
                test_files_by_video[video_name] = []
                total_train_frames += 1
                total_val_frames += 1
            else:
                # Normal splitting for 3+ frames
                if test_split > 0:
                    train_val_files, test_files = train_test_split(
                        video_keys, test_size=test_split, random_state=random_seed
                    )
                    if len(train_val_files) >= 2:
                        relative_val_ratio = val_split / (1 - test_split)
                        train_files, val_files = train_test_split(
                            train_val_files, test_size=relative_val_ratio, random_state=random_seed
                        )
                    else:
                        train_files = train_val_files
                        val_files = []
                    test_files_by_video[video_name] = test_files
                else:
                    train_files, val_files = train_test_split(
                        video_keys, test_size=val_split, random_state=random_seed
                    )
                    test_files_by_video[video_name] = []
                
                train_files_by_video[video_name] = train_files
                val_files_by_video[video_name] = val_files
                
                total_train_frames += len(train_files)
                total_val_frames += len(val_files)
                total_test_frames += len(test_files_by_video[video_name])
            
            print(f"  {video_name}: {len(train_files_by_video[video_name])} train, "
                  f"{len(val_files_by_video[video_name])} val, "
                  f"{len(test_files_by_video[video_name])} test frames")
        
        print(f"\nTotal frames split: {total_train_frames} train, {total_val_frames} val, {total_test_frames} test")
        
        # Create dataset directories with video subfolders
        splits = [('train', train_files_by_video), ('val', val_files_by_video)]
        if any(test_files_by_video.values()):
            splits.append(('test', test_files_by_video))
        
        for split_name, files_by_video in splits:
            for video_name, video_keys in files_by_video.items():
                if not video_keys:  # Skip if no files for this split
                    continue
                    
                # Create video-specific directories
                video_images_dir = output_dir / 'images' / split_name / video_name
                video_labels_dir = output_dir / 'labels' / split_name / video_name
                video_images_dir.mkdir(parents=True, exist_ok=True)
                video_labels_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"Extracting {len(video_keys)} frames for {video_name} ({split_name})")
                
                # Process all frames for this video in this split with sequential numbering
                successful_extractions = 0
                for idx, key in enumerate(video_keys, 1):
                    annotation_data = all_annotations[key]
                    frame_number = annotation_data['frame_number']
                    video_path = annotation_data['video_path']
                    content = annotation_data['content']
                    
                    # Use sequential numbering: 1.txt, 2.txt, etc.
                    label_filename = f"{idx}.txt"
                    label_path = video_labels_dir / label_filename
                    
                    # Write annotation file
                    with open(label_path, 'w') as f:
                        f.write(content)
                    
                    # Extract actual frame from video: 1.jpg, 2.jpg, etc.
                    image_filename = f"{idx}.jpg"
                    image_path = video_images_dir / image_filename
                    
                    if extract_frame_from_video(video_path, frame_number, image_path):
                        successful_extractions += 1
                    else:
                        # If frame extraction fails, create empty placeholder
                        image_path.touch()
                
                print(f"  Successfully extracted {successful_extractions}/{len(video_keys)} frames")
        
        # Create YOLO yaml file
        create_yolo_yaml(output_dir, label_mapping)
        
        print(f"Dataset created successfully at {output_dir}")
        print(f"Extracted actual frames from videos based on COCO annotations.")
        
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