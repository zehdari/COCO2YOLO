import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QSpinBox,
    QDoubleSpinBox, QTextEdit, QGroupBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextBrowser, 
    QTabWidget, QCheckBox, QDialog, QDialogButtonBox, QListWidget,
    QListWidgetItem, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import json
from dataclasses import dataclass, field
import shutil
import os
import markdown
import yaml
from typing import Dict, Optional, List, Tuple, Set
from sklearn.model_selection import train_test_split
import re

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        # If the app is run as a bundled executable, _MEIPASS will exist.
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        # Otherwise, use the current working directory (or adjust as needed)
        base_path = Path(os.path.abspath("."))
    return base_path / relative_path

@dataclass
class FrameRange:
    """Represents a range of frames to include"""
    start: int
    end: int
    
    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(f"Start frame {self.start} cannot be greater than end frame {self.end}")
    
    def contains(self, frame_num: int) -> bool:
        """Check if a frame number is within this range"""
        return self.start <= frame_num <= self.end
    
    def __str__(self):
        return f"{self.start}-{self.end}"

@dataclass
class FolderConfig:
    """Configuration for a specific folder"""
    folder_path: Path
    enabled: bool = True
    frame_ranges: List[FrameRange] = field(default_factory=list)
    
    def should_include_frame(self, frame_num: int) -> bool:
        """Check if a frame should be included based on the configured ranges"""
        if not self.frame_ranges:  # If no ranges specified, include all frames
            return True
        return any(range.contains(frame_num) for range in self.frame_ranges)
    
    @staticmethod
    def parse_frame_ranges(range_string: str) -> List[FrameRange]:
        """Parse frame ranges from string like '0-280, 543-564, 668-679'"""
        if not range_string.strip():
            return []
        
        ranges = []
        for part in range_string.split(','):
            part = part.strip()
            if '-' in part:
                try:
                    start, end = map(int, part.split('-', 1))
                    ranges.append(FrameRange(start, end))
                except ValueError:
                    raise ValueError(f"Invalid range format: '{part}'. Expected format: 'start-end'")
            elif part.isdigit():
                # Single frame
                frame = int(part)
                ranges.append(FrameRange(frame, frame))
            else:
                raise ValueError(f"Invalid range format: '{part}'. Expected format: 'start-end' or single number")
        
        return ranges

class FrameRangeDialog(QDialog):
    """Dialog for configuring frame ranges for folders"""
    
    def __init__(self, folder_configs: List[FolderConfig], parent=None):
        super().__init__(parent)
        self.folder_configs = folder_configs
        self.setWindowTitle("Configure Frame Ranges")
        self.setMinimumSize(600, 400)
        self.setupUI()
        
    def setupUI(self):
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "Configure frame ranges for each folder. Leave empty to include all frames.\n"
            "Format: '0-280, 543-564, 668-679' or single frames: '100, 200, 300'"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Table for folder configurations
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Folder", "Enabled", "Frame Ranges"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        
        self.populate_table()
        layout.addWidget(self.table)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def populate_table(self):
        """Populate the table with folder configurations"""
        self.table.setRowCount(len(self.folder_configs))
        
        for row, config in enumerate(self.folder_configs):
            # Folder name
            folder_item = QTableWidgetItem(config.folder_path.name)
            folder_item.setFlags(folder_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, folder_item)
            
            # Enabled checkbox
            enabled_item = QTableWidgetItem()
            enabled_item.setCheckState(Qt.CheckState.Checked if config.enabled else Qt.CheckState.Unchecked)
            enabled_item.setFlags(enabled_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, enabled_item)
            
            # Frame ranges
            ranges_text = ", ".join(str(r) for r in config.frame_ranges)
            ranges_item = QTableWidgetItem(ranges_text)
            self.table.setItem(row, 2, ranges_item)
    
    def get_configurations(self) -> List[FolderConfig]:
        """Get the updated configurations from the table"""
        updated_configs = []
        
        for row in range(self.table.rowCount()):
            config = self.folder_configs[row]
            
            # Update enabled status
            enabled_item = self.table.item(row, 1)
            config.enabled = enabled_item.checkState() == Qt.CheckState.Checked
            
            # Update frame ranges
            ranges_item = self.table.item(row, 2)
            ranges_text = ranges_item.text() if ranges_item else ""
            
            try:
                config.frame_ranges = FolderConfig.parse_frame_ranges(ranges_text)
            except ValueError as e:
                raise ValueError(f"Error in folder '{config.folder_path.name}': {str(e)}")
            
            updated_configs.append(config)
        
        return updated_configs

def create_or_update_yolo_yaml(
    dataset_base: Path,
    label_mapping: Dict[str, str],
    output_yaml_path: Path,
):
    """
    Creates (or updates) a YOLO data.yaml file listing all train/val subfolders.
    """
    # --- 1) Gather subfolders in train/val (and optionally test) ---
    train_dir = dataset_base / "images" / "train"
    val_dir   = dataset_base / "images" / "val"
    test_dir  = dataset_base / "images" / "test"  # if you want to include test

    def list_subfolders(dir_path: Path) -> List[str]:
        """Return list of subfolder paths (as strings) for a given directory."""
        subfolders = []
        if dir_path.is_dir():
            for item in sorted(dir_path.iterdir()):
                if item.is_dir():
                    subfolders.append(str(item.resolve()))
        return subfolders

    train_subfolders = list_subfolders(train_dir)
    val_subfolders   = list_subfolders(val_dir)
    test_subfolders  = list_subfolders(test_dir)  # only if needed

    # --- 2) Load existing YAML (if any) and decide how to handle merges/overwrites ---
    data = {}
    if output_yaml_path.exists():
        with open(output_yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

    # Always store nc and names fresh:
    data["nc"] = len(label_mapping)
    data["names"] = list(label_mapping.keys())  

    # Prepare the new train/val/test lists and merge with existing if present
    old_train = data.get("train", [])
    old_val   = data.get("val", [])
    old_test  = data.get("test", [])

    # Handle existing YAML entries that might be a single string or list:
    if isinstance(old_train, str):
        old_train = [old_train]
    if isinstance(old_val, str):
        old_val = [old_val]
    if isinstance(old_test, str):
        old_test = [old_test]

    # Combine old + new, ensure uniqueness
    data["train"] = sorted(set(old_train + train_subfolders))
    data["val"]   = sorted(set(old_val + val_subfolders))
    if test_subfolders:
        data["test"] = sorted(set(old_test + test_subfolders))
    elif "test" in data:
        # If you prefer removing 'test' if there's no test folders
        del data["test"]

    # Write the final data back to YAML in bracketed style
    with open(output_yaml_path, "w") as f:
        # Write `train:` in bracketed multi-line style
        if "train" in data and data["train"]:
            train_str = ",\n  ".join(data["train"])
            f.write(f"train: [\n  {train_str}\n]\n")
        else:
            f.write("train: []\n")

        # Write `val:` in bracketed multi-line style
        if "val" in data and data["val"]:
            val_str = ",\n  ".join(data["val"])
            f.write(f"val: [\n  {val_str}\n]\n")
        else:
            f.write("val: []\n")

        # (Optional) `test:` in bracketed multi-line style
        if "test" in data and data["test"]:
            test_str = ",\n  ".join(data["test"])
            f.write(f"test: [\n  {test_str}\n]\n")

        # Number of classes
        f.write(f"nc: {data['nc']}\n")

        # names: bracketed, with single quotes around each category name
        quoted_names = ", ".join(f"'{name}'" for name in data["names"])
        f.write(f"names: [{quoted_names}]\n")


@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    input_path: Path
    dataset_base: Path
    val_split_ratio: float = 0.2
    test_split_ratio: float = 0.0
    label_mapping: Dict[str, str] = None
    random_seed: int = 42
    rename_files: bool = True
    dataset_name: Optional[str] = None
    create_yaml: bool = True
    folder_configs: List[FolderConfig] = field(default_factory=list)

    def __post_init__(self):
        self.input_path = Path(self.input_path)
        self.dataset_base = Path(self.dataset_base)
        
        if self.dataset_name is None:
            self.dataset_name = self.input_path.name
        
        if self.label_mapping is None:
            raise ValueError("label_mapping must be provided - cannot be None")

class COCOtoYOLOConverter:
    """Handles conversion of COCO format annotations to YOLO format"""
    
    def __init__(self, config: DatasetConfig, progress_signal):
        self.config = config
        self.progress_signal = progress_signal
        self.file_mapping = {}

    def is_cvat_coco_folder(self, folder: Path) -> bool:
        """Check if a folder has the CVAT COCO structure"""
        images_dir = folder / 'images'
        annotations_dir = folder / 'annotations'
        return (
            folder.is_dir() and 
            images_dir.exists() and 
            annotations_dir.exists() and 
            any(annotations_dir.glob('*.json'))
        )

    def find_cvat_coco_folders(self, parent_dir: Path) -> List[Path]:
        """
        Walk through the entire directory tree rooted at parent_dir and return a list
        of valid CVAT COCO folders. A folder is considered valid if self.is_cvat_coco_folder
        returns True.
        """
        cvat_folders = set()

        # os.walk will yield the parent_dir as the first iteration, then all subdirectories
        for dirpath, _dirnames, _filenames in os.walk(parent_dir):
            path = Path(dirpath)
            if self.is_cvat_coco_folder(path):
                cvat_folders.add(path)

        # Return the unique folders in sorted order
        return sorted(cvat_folders)

    def setup_directories(self) -> None:
        """Create necessary directories based on configuration"""
        dataset_name = self.config.dataset_name

        # Always use a temporary folder inside dataset_base
        temp_dir = self.config.dataset_base / "__temp__" / dataset_name
        self.label_output_dir = temp_dir / "labels"
        self.image_output_dir = temp_dir / "images"
        self.label_output_dir.mkdir(parents=True, exist_ok=True)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def coco_to_yolo_polygon(coco_polygon: List[float], img_width: int, img_height: int) -> List[Tuple[float, float]]:
        """Convert COCO polygon coordinates to YOLO format"""
        return [
            (coco_polygon[i] / img_width, coco_polygon[i+1] / img_height) 
            for i in range(0, len(coco_polygon), 2)
        ]

    def log(self, message: str) -> None:
        """Emit a log message through the progress signal"""
        self.progress_signal.emit(message)

    def get_frame_number_from_filename(self, filename: str) -> Optional[int]:
        """Extract frame number from filename. Assumes format like 'frame_001.jpg' or '001.jpg'"""
        # Try different patterns to extract frame numbers
        patterns = [
            r'frame_(\d+)',  # frame_001.jpg
            r'(\d+)',        # 001.jpg or just numbers
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None

    def convert_annotations(self, json_file_path: Path, folder_config: FolderConfig) -> None:
        """Convert COCO JSON annotations to YOLO format with frame filtering"""
        try:
            with open(json_file_path) as f:
                data = json.load(f)

            category_mapping = {cat['id']: cat['name'] for cat in data['categories']}
            
            processed_count = 0
            filtered_count = 0
            
            for index, image in enumerate(data['images']):
                # Extract frame number from filename
                frame_num = self.get_frame_number_from_filename(image['file_name'])
                
                # Check if this frame should be included
                if frame_num is not None and not folder_config.should_include_frame(frame_num):
                    filtered_count += 1
                    continue
                
                image_id = image['id']
                img_width, img_height = image['width'], image['height']
                annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
                
                output_filename = (
                    f'{processed_count}.txt' if self.config.rename_files 
                    else Path(image['file_name']).with_suffix('.txt').name
                )
                
                # Map original image name to the new one
                new_image_name = (
                    f"{processed_count}{Path(image['file_name']).suffix}"
                    if self.config.rename_files
                    else image['file_name']
                )
                self.file_mapping[image['file_name']] = new_image_name
                
                output_file = self.label_output_dir / output_filename
                with open(output_file, 'w') as f_lbl:
                    for ann in annotations:
                        category_id = ann['category_id']
                        for coco_polygon in ann['segmentation']:
                            yolo_polygon = self.coco_to_yolo_polygon(
                                coco_polygon, img_width, img_height
                            )
                            line = (
                                f"{self.config.label_mapping[category_mapping[category_id]]} " +
                                " ".join(f"{x} {y}" for x, y in yolo_polygon)
                            )
                            f_lbl.write(line + '\n')
                
                processed_count += 1
            
            if filtered_count > 0:
                self.log(f"Filtered out {filtered_count} frames based on frame ranges")
            self.log(f"Processed {processed_count} frames from annotations")
            
        except Exception as e:
            self.log(f"Error processing annotations: {str(e)}")
            raise

    def check_missing_mappings(self, json_file_path: Path) -> Set[str]:
        """Scan actual annotations and identify categories missing from label_mapping"""
        try:
            with open(json_file_path) as f:
                data = json.load(f)
                
            category_mapping = {cat['id']: cat['name'] for cat in data['categories']}
            used_category_ids = {ann['category_id'] for ann in data['annotations']}
            used_categories = {category_mapping[cat_id] for cat_id in used_category_ids}
            missing_categories = used_categories - set(self.config.label_mapping.keys())
            
            if missing_categories:
                self.log("\nWARNING: Found categories in annotations without label mappings:")
                for cat in sorted(missing_categories):
                    self.log(f"  - {cat}")
                self.log("\nPlease add these categories to your label_mapping dictionary.")
                
                self.log("\nCategory usage statistics:")
                for cat_id in used_category_ids:
                    cat_name = category_mapping[cat_id]
                    count = sum(1 for ann in data['annotations'] if ann['category_id'] == cat_id)
                    self.log(f"  - {cat_name}: {count} instances")
                
            return missing_categories
                
        except Exception as e:
            self.log(f"Error checking label mappings: {str(e)}")
            raise
    
    def find_images(self, folder: Path) -> List[Path]:
        """Find all image files in a folder"""
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for pattern in patterns:
            images.extend(list(folder.glob(pattern)))
        return sorted(images)

    def process_empty_annotations(self, images_folder: Path, folder_config: FolderConfig) -> None:
        """Create empty annotation files for images without annotations, with frame filtering"""
        image_files = self.find_images(images_folder)
        
        if not image_files:
            raise ValueError(f"No image files found in {images_folder}")
        
        processed_count = 0
        filtered_count = 0
        
        for image_file in image_files:
            # Extract frame number from filename
            frame_num = self.get_frame_number_from_filename(image_file.name)
            
            # Check if this frame should be included
            if frame_num is not None and not folder_config.should_include_frame(frame_num):
                filtered_count += 1
                continue
            
            output_filename = (
                f'{processed_count}.txt' if self.config.rename_files 
                else image_file.with_suffix('.txt').name
            )
            new_image_name = (
                f"{processed_count}{image_file.suffix}"
                if self.config.rename_files
                else image_file.name
            )
            self.file_mapping[image_file.name] = new_image_name

            (self.label_output_dir / output_filename).touch()
            processed_count += 1
        
        if filtered_count > 0:
            self.log(f"Filtered out {filtered_count} images based on frame ranges")
        self.log(f"Created {processed_count} empty annotation files")

    def copy_and_rename_images(self, source_folder: Path, folder_config: FolderConfig) -> None:
        """Copy and optionally rename images to the temporary output folder, with frame filtering"""
        files = self.find_images(source_folder)
        
        if not files:
            raise ValueError(f"No image files found in {source_folder}")
        
        processed_count = 0
        filtered_count = 0
        
        for file_path in files:
            # Extract frame number from filename
            frame_num = self.get_frame_number_from_filename(file_path.name)
            
            # Check if this frame should be included
            if frame_num is not None and not folder_config.should_include_frame(frame_num):
                filtered_count += 1
                continue
            
            if self.config.rename_files:
                new_filename = f"{processed_count}{file_path.suffix.lower()}"
            else:
                new_filename = file_path.name
                
            new_path = self.image_output_dir / new_filename
            shutil.copy2(file_path, new_path)
            processed_count += 1
        
        if filtered_count > 0:
            self.log(f"Filtered out {filtered_count} images based on frame ranges")
        self.log(f"Copied {processed_count} images")

    def split_and_organize_dataset(self) -> None:
        """Split dataset into train, val (and optional test) sets and organize files."""
        try:
            dataset_name = self.config.dataset_name
            # Final destination directories:
            train_image_dir = self.config.dataset_base / 'images' / 'train' / dataset_name
            val_image_dir   = self.config.dataset_base / 'images' / 'val' / dataset_name
            train_label_dir = self.config.dataset_base / 'labels' / 'train' / dataset_name
            val_label_dir   = self.config.dataset_base / 'labels' / 'val' / dataset_name

            test_image_dir = None
            test_label_dir = None
            if self.config.test_split_ratio > 0:
                test_image_dir = self.config.dataset_base / 'images' / 'test' / dataset_name
                test_label_dir = self.config.dataset_base / 'labels' / 'test' / dataset_name

            # Collect files from the temporary directories
            image_files = sorted(os.listdir(self.image_output_dir))
            label_files = sorted(os.listdir(self.label_output_dir))
            
            if not image_files:
                raise ValueError("No image files found in output directory")
            if not label_files:
                raise ValueError("No label files found in output directory")
                
            if len(image_files) != len(label_files):
                raise ValueError(
                    f"Mismatch between number of images ({len(image_files)}) "
                    f"and labels ({len(label_files)})"
                )
            
            self.log(
                f"Splitting {len(image_files)} files into train/val sets for dataset '{dataset_name}'..."
            )
            
            total_split_ratio = self.config.val_split_ratio + self.config.test_split_ratio
            if total_split_ratio >= 1:
                raise ValueError("Combined validation and test split ratios must be less than 1.0")
                
            # Check we have enough samples to split
            min_ratio = min(
                r for r in [self.config.val_split_ratio, self.config.test_split_ratio]
                if r > 0
            ) if (self.config.val_split_ratio > 0 or self.config.test_split_ratio > 0) else 0
            if min_ratio > 0:
                min_samples = max(2, int(1 / min_ratio))
                if len(image_files) < min_samples:
                    raise ValueError(
                        f"Not enough samples for split. Need at least {min_samples} files, "
                        f"but only found {len(image_files)}"
                    )
            
            # Perform train/val/test split
            if self.config.test_split_ratio > 0:
                # First split out the test set
                remaining_ratio = 1 - self.config.test_split_ratio
                relative_val_ratio = (
                    self.config.val_split_ratio / remaining_ratio
                    if remaining_ratio > 0
                    else 0
                )
                
                train_val_images, test_images = train_test_split(
                    image_files,
                    test_size=self.config.test_split_ratio,
                    random_state=self.config.random_seed
                )
                train_val_labels, test_labels = train_test_split(
                    label_files,
                    test_size=self.config.test_split_ratio,
                    random_state=self.config.random_seed
                )
                
                # Then split train vs val from the remainder
                train_images, val_images = train_test_split(
                    train_val_images,
                    test_size=relative_val_ratio,
                    random_state=self.config.random_seed
                )
                train_labels, val_labels = train_test_split(
                    train_val_labels,
                    test_size=relative_val_ratio,
                    random_state=self.config.random_seed
                )
            else:
                train_images, val_images = train_test_split(
                    image_files,
                    test_size=self.config.val_split_ratio,
                    random_state=self.config.random_seed
                )
                train_labels, val_labels = train_test_split(
                    label_files,
                    test_size=self.config.val_split_ratio,
                    random_state=self.config.random_seed
                )
                test_images = []
                test_labels = []
            
            self.log(
                f"Split complete for '{dataset_name}': "
                f"{len(train_images)} training files, "
                f"{len(val_images)} validation files"
                + (f", {len(test_images)} test files" if self.config.test_split_ratio > 0 else "")
            )

            # Create final directories if they don't exist
            train_image_dir.mkdir(parents=True, exist_ok=True)
            val_image_dir.mkdir(parents=True, exist_ok=True)
            train_label_dir.mkdir(parents=True, exist_ok=True)
            val_label_dir.mkdir(parents=True, exist_ok=True)
            if test_image_dir and test_label_dir:
                test_image_dir.mkdir(parents=True, exist_ok=True)
                test_label_dir.mkdir(parents=True, exist_ok=True)

            # Helper to move files from the temporary location to the final subset directories
            def move_files(file_list, source_dir, dest_dir):
                for fn in file_list:
                    src_file = source_dir / fn
                    dst_file = dest_dir / fn
                    if not dst_file.exists():
                        shutil.move(str(src_file), str(dst_file))

            # Move images
            move_files(train_images, self.image_output_dir, train_image_dir)
            move_files(val_images,   self.image_output_dir, val_image_dir)
            if test_image_dir:
                move_files(test_images, self.image_output_dir, test_image_dir)

            # Move labels
            move_files(train_labels, self.label_output_dir, train_label_dir)
            move_files(val_labels,   self.label_output_dir, val_label_dir)
            if test_label_dir:
                move_files(test_labels, self.label_output_dir, test_label_dir)

            self.log(f"Dataset organization completed successfully for '{dataset_name}'!")

            if self.config.create_yaml:
                output_yaml_path = self.config.dataset_base / "data.yaml"
                create_or_update_yolo_yaml(
                    dataset_base=self.config.dataset_base,
                    label_mapping=self.config.label_mapping,
                    output_yaml_path=output_yaml_path,
                )
                self.log(f"YOLO data.yaml updated/created at {output_yaml_path}")
            else:
                self.log("Skipping YAML updating/creation")

        except Exception as e:
            print(f"Error during dataset organization: {str(e)}")
            raise

    def process(self) -> None:
        """Main processing function"""
        # Get enabled folder configs
        enabled_configs = [config for config in self.config.folder_configs if config.enabled]
        
        if not enabled_configs:
            raise ValueError("No folders are enabled for processing")
        
        self.log(f"Processing {len(enabled_configs)} enabled folder(s):")
        for config in enabled_configs:
            range_info = f" (frames: {', '.join(str(r) for r in config.frame_ranges)})" if config.frame_ranges else " (all frames)"
            self.log(f"  - {config.folder_path.name}{range_info}")

        # Process each enabled folder
        for folder_config in enabled_configs:
            try:
                self.log(f"\nProcessing folder: {folder_config.folder_path.name}")
                
                # Create a new config for this folder
                single_folder_config = DatasetConfig(
                    input_path=folder_config.folder_path,
                    dataset_base=self.config.dataset_base,
                    val_split_ratio=self.config.val_split_ratio,
                    test_split_ratio=self.config.test_split_ratio,
                    label_mapping=self.config.label_mapping,
                    random_seed=self.config.random_seed,
                    rename_files=self.config.rename_files,
                    dataset_name=folder_config.folder_path.name,
                    create_yaml=self.config.create_yaml
                )
                
                # Create a new converter instance for this folder
                folder_converter = COCOtoYOLOConverter(single_folder_config, self.progress_signal)
                folder_converter.process_single_folder(single_folder_config, folder_config)
                
            except Exception as e:
                self.log(f"Error processing folder {folder_config.folder_path.name}: {str(e)}")
                # Continue with next folder instead of stopping
                continue

    def process_single_folder(self, config: DatasetConfig, folder_config: FolderConfig) -> None:
        """Process a single CVAT COCO folder with frame filtering"""
        images_folder = config.input_path / 'images'
        annotations_folder = config.input_path / 'annotations'

        # Validate input paths
        if not images_folder.exists():
            raise FileNotFoundError(f"Images directory not found: {images_folder}")

        # Create temporary output directories for images/labels
        self.setup_directories()

        # Check for images before proceeding
        image_files = self.find_images(images_folder)
        if not image_files:
            raise ValueError(f"No image files (jpg, jpeg, png) found in {images_folder}")
        
        self.log(f"Found {len(image_files)} images in {images_folder}")

        try:
            # Process annotations if they exist
            if annotations_folder.exists():
                json_files = list(annotations_folder.glob('*.json'))
                if json_files:
                    self.log(f"Found annotation file: {json_files[0].name}")
                    
                    # Check for missing mappings before proceeding with conversion
                    missing_categories = self.check_missing_mappings(json_files[0])
                    if missing_categories:
                        raise ValueError("Missing category mappings must be addressed before conversion")
                        
                    self.convert_annotations(json_files[0], folder_config)
                else:
                    self.log(f"No JSON files found in {annotations_folder}, creating empty annotations")
                    self.process_empty_annotations(images_folder, folder_config)
            else:
                self.log("No annotations folder found, creating empty annotations")
                self.process_empty_annotations(images_folder, folder_config)

            # Copy and rename images
            self.copy_and_rename_images(images_folder, folder_config)
            
            # Ensure we have files to organize
            if not os.path.exists(self.image_output_dir) or not os.listdir(self.image_output_dir):
                raise ValueError("No images were processed successfully")
            if not os.path.exists(self.label_output_dir) or not os.listdir(self.label_output_dir):
                raise ValueError("No labels were generated successfully")
                
            # Split / organize
            self.split_and_organize_dataset()
            
        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
            raise
        finally:
            # Clean up the temporary directory
            temp_dir = config.dataset_base / "__temp__" / config.dataset_name
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # If __temp__ is now empty, remove it
            parent_temp = config.dataset_base / "__temp__"
            if parent_temp.exists():
                try:
                    parent_temp.rmdir()  # Will only succeed if empty
                except OSError:
                    pass


class ConversionWorker(QThread):
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            # Pass the progress signal to the converter
            converter = COCOtoYOLOConverter(self.config, self.progress)
            converter.process()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class LabelMappingTable(QTableWidget):
    """Custom table widget for managing label mappings"""
    def __init__(self):
        super().__init__()
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Category Name", "YOLO ID"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(1, 100)

    def get_mapping(self):
        """Get the current mapping as a dictionary"""
        mapping = {}
        for row in range(self.rowCount()):
            category_item = self.item(row, 0)
            yolo_id_item = self.item(row, 1)
            if category_item and yolo_id_item:
                category = category_item.text().strip()
                yolo_id = yolo_id_item.text().strip()
                if yolo_id:  # Only include if YOLO ID is specified
                    mapping[category] = yolo_id
        return mapping

    def set_mapping(self, mapping):
        """Set the mapping from a dictionary"""
        self.setRowCount(len(mapping))
        for row, (category, yolo_id) in enumerate(mapping.items()):
            self.setItem(row, 0, QTableWidgetItem(category))
            self.setItem(row, 1, QTableWidgetItem(str(yolo_id)))

    def load_categories(self, categories):
        """Load COCO categories into the table"""
        self.setRowCount(len(categories))
        for row, category in enumerate(categories):
            self.setItem(row, 0, QTableWidgetItem(category))
            self.setItem(row, 1, QTableWidgetItem(str(row)))  # Default sequential numbering

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COCO to YOLO Converter")
        self.setMinimumWidth(1200)
        self.folder_configs = []
        self.setupUI()
        
    def setupUI(self):
        # Create a QTabWidget to hold multiple tabs (Conversion and Help)
        tab_widget = QTabWidget()
        self.setCentralWidget(tab_widget)
        
        # ----- Conversion Tab -----
        conversion_tab = QWidget()
        conversion_layout = QHBoxLayout(conversion_tab)
        
        # Create splitter for better layout control
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side container for groups (Path Config, Configuration, Folder Selection, Log, Convert Button)
        left_widget = QWidget()
        left_container = QVBoxLayout(left_widget)
        
        # Path configuration group
        path_group = QGroupBox("Path Configuration")
        path_layout = QVBoxLayout()
        # Input path
        input_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText(
            "Select parent folder containing COCO folders"
        )
        self.input_path.textChanged.connect(self.input_path_changed)
        self.input_path.textChanged.connect(self.update_convert_button_state)
        input_button = QPushButton("Browse...")
        input_button.clicked.connect(lambda: self.browse_folder(self.input_path))
        input_layout.addWidget(QLabel("Input Path:"))
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(input_button)
        path_layout.addLayout(input_layout)
        # Dataset base path
        dataset_layout = QHBoxLayout()
        self.dataset_base = QLineEdit()
        self.dataset_base.textChanged.connect(self.update_convert_button_state)
        self.dataset_button = QPushButton("Browse...")
        self.dataset_button.clicked.connect(lambda: self.browse_folder(self.dataset_base))
        dataset_layout.addWidget(QLabel("Dataset Base:"))
        dataset_layout.addWidget(self.dataset_base)
        dataset_layout.addWidget(self.dataset_button)
        path_layout.addLayout(dataset_layout)
        path_group.setLayout(path_layout)
        left_container.addWidget(path_group)
        
        # Configuration group (Split ratios, Random Seed)
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        # Validation split ratio
        val_layout = QHBoxLayout()
        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.0, 1.0)
        self.val_split.setSingleStep(0.1)
        self.val_split.setValue(0.2)
        val_layout.addWidget(QLabel("Validation Split Ratio:"))
        val_layout.addWidget(self.val_split)
        config_layout.addLayout(val_layout)
        # Test split ratio
        test_layout = QHBoxLayout()
        self.test_split = QDoubleSpinBox()
        self.test_split.setRange(0.0, 1.0)
        self.test_split.setSingleStep(0.1)
        self.test_split.setValue(0.0)
        test_layout.addWidget(QLabel("Test Split Ratio:"))
        test_layout.addWidget(self.test_split)
        config_layout.addLayout(test_layout)
        # Random seed
        seed_layout = QHBoxLayout()
        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 99999)
        self.random_seed.setValue(42)
        seed_layout.addWidget(QLabel("Random Seed:"))
        seed_layout.addWidget(self.random_seed)
        config_layout.addLayout(seed_layout)
        config_group.setLayout(config_layout)
        left_container.addWidget(config_group)
        
        # Folder Selection Group
        folder_group = QGroupBox("Folder Selection & Frame Ranges")
        folder_layout = QVBoxLayout()
        
        # Folder list
        self.folder_list = QListWidget()
        self.folder_list.setMaximumHeight(150)
        folder_layout.addWidget(self.folder_list)
        
        # Configure ranges button
        self.configure_ranges_button = QPushButton("Configure Frame Ranges...")
        self.configure_ranges_button.setEnabled(False)
        self.configure_ranges_button.clicked.connect(self.configure_frame_ranges)
        folder_layout.addWidget(self.configure_ranges_button)
        
        folder_group.setLayout(folder_layout)
        left_container.addWidget(folder_group)
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        left_container.addWidget(self.log_output)
        
        # Convert button and YAML toggle
        button_layout = QVBoxLayout()
        self.create_yaml_toggle = QCheckBox("Create/Update data.yaml")
        self.create_yaml_toggle.setChecked(True)
        button_layout.addWidget(self.create_yaml_toggle)
        
        self.convert_button = QPushButton("Convert")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self.start_conversion)
        button_layout.addWidget(self.convert_button)
        
        left_container.addLayout(button_layout)
        
        # Right side - Label Mapping
        mapping_group = QGroupBox("Label Mapping")
        mapping_layout = QVBoxLayout()
        self.mapping_table = LabelMappingTable()
        mapping_layout.addWidget(self.mapping_table)
        mapping_group.setLayout(mapping_layout)
        
        # Add widgets to splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(mapping_group)
        main_splitter.setSizes([700, 500])  # Set initial sizes
        
        conversion_layout.addWidget(main_splitter)
        
        # Add the Conversion tab to the tab widget
        tab_widget.addTab(conversion_tab, "Conversion")
        
        # ----- Help Tab (Markdown Viewer) -----
        help_tab = QWidget()
        help_layout = QVBoxLayout(help_tab)
        self.markdown_viewer = QTextBrowser()
        help_layout.addWidget(self.markdown_viewer)
        tab_widget.addTab(help_tab, "Help")

        # Load markdown text from a file (e.g., help.md)
        help_file = resource_path("Resources/help.md")
        if help_file.exists():
            try:
                with open(help_file, "r", encoding="utf-8") as f:
                    markdown_text = f.read()
            except Exception as e:
                markdown_text = f"# Error Loading Help File\n\n{str(e)}"
        else:
            markdown_text = self.get_default_help_text()

        # Convert markdown to HTML and set the content of the viewer
        try:
            html = markdown.markdown(markdown_text)
            self.markdown_viewer.setHtml(html)
        except Exception as e:
            # Fallback: display plain text if conversion fails
            self.markdown_viewer.setPlainText(markdown_text)

    def get_default_help_text(self):
        """Return default help text if help.md is not found"""
        return """# COCO to YOLO Converter with Frame Range Selection

## Overview
This tool converts COCO format annotations to YOLO format with support for selective frame processing.

## Features
- **Frame Range Selection**: Process specific frame ranges from each folder
- **Batch Processing**: Handle multiple COCO folders at once
- **Train/Val/Test Splitting**: Automatically organize data into training sets
- **Label Mapping**: Map COCO categories to YOLO class IDs

## Frame Range Format
When configuring frame ranges, use the following formats:
- Single frames: `100, 200, 300`
- Frame ranges: `0-280, 543-564, 668-679`
- Mixed: `0-100, 150, 200-300`

## Usage
1. Select input folder containing COCO datasets
2. Choose output dataset base directory
3. Configure frame ranges for each folder (optional)
4. Set up label mappings
5. Click Convert

## Frame Number Detection
The tool automatically detects frame numbers from filenames using patterns like:
- `frame_001.jpg`
- `001.jpg`
- Any filename containing numbers

If no frame ranges are specified, all frames will be processed.
"""

    def input_path_changed(self, new_path):
        """Handle input path changes and discover COCO folders"""
        if not new_path:
            self.folder_configs.clear()
            self.update_folder_list()
            return
            
        path = Path(new_path)
        if not path.exists():
            return
        
        # Create a temporary converter to find COCO folders
        temp_config = DatasetConfig(
            input_path=path,
            dataset_base=Path("/tmp"),  # Dummy path
            label_mapping={"dummy": "0"}  # Dummy mapping
        )
        temp_converter = COCOtoYOLOConverter(temp_config, lambda x: None)
        
        # Find all COCO folders
        coco_folders = temp_converter.find_cvat_coco_folders(path)
        
        # Create folder configs
        self.folder_configs = [FolderConfig(folder_path=folder) for folder in coco_folders]
        
        # Update UI
        self.update_folder_list()
        self.configure_ranges_button.setEnabled(len(self.folder_configs) > 0)
        
        # Load categories from all folders
        self.load_categories_from_folders()

    def update_folder_list(self):
        """Update the folder list display"""
        self.folder_list.clear()
        for config in self.folder_configs:
            item = QListWidgetItem()
            enabled_text = "✓" if config.enabled else "✗"
            range_text = f" ({', '.join(str(r) for r in config.frame_ranges)})" if config.frame_ranges else ""
            item.setText(f"{enabled_text} {config.folder_path.name}{range_text}")
            self.folder_list.addItem(item)

    def load_categories_from_folders(self):
        """Load categories from all discovered COCO folders"""
        self.mapping_table.setRowCount(0)
        
        all_categories = set()
        for config in self.folder_configs:
            json_files = list((config.folder_path / "annotations").glob("*.json"))
            if json_files:
                try:
                    with open(json_files[0]) as f:
                        data = json.load(f)
                    categories = {cat['name'] for cat in data['categories']}
                    all_categories.update(categories)
                except Exception as e:
                    self.log_message(f"Error loading categories from {config.folder_path.name}: {str(e)}")
        
        if all_categories:
            sorted_categories = sorted(all_categories)
            self.mapping_table.setRowCount(len(sorted_categories))
            for i, category in enumerate(sorted_categories):
                self.mapping_table.setItem(i, 0, QTableWidgetItem(category))
                self.mapping_table.setItem(i, 1, QTableWidgetItem(str(i)))
            
            self.log_message(f"Loaded {len(sorted_categories)} unique categories from {len(self.folder_configs)} folders")

    def configure_frame_ranges(self):
        """Open the frame range configuration dialog"""
        dialog = FrameRangeDialog(self.folder_configs, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.folder_configs = dialog.get_configurations()
                self.update_folder_list()
                self.log_message("Frame range configuration updated")
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Configuration", str(e))

    def update_convert_button_state(self):
        """Update the convert button state based on inputs"""
        have_input_path = bool(self.input_path.text().strip())
        have_dataset_base = bool(self.dataset_base.text().strip())
        have_folders = len(self.folder_configs) > 0
        can_convert = have_input_path and have_dataset_base and have_folders
        self.convert_button.setEnabled(can_convert)

    def browse_folder(self, line_edit):
        """Browse for a folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit.setText(folder)

    def log_message(self, message):
        """Add a message to the log"""
        self.log_output.append(message)

    def show_error(self, message):
        """Show an error message"""
        QMessageBox.critical(self, "Error", message)
        self.convert_button.setEnabled(True)

    def conversion_finished(self):
        """Handle conversion completion"""
        self.log_message("Conversion completed successfully!")
        self.convert_button.setEnabled(True)

    def start_conversion(self):
        """Start the conversion process"""
        try:
            # Validate inputs
            if not self.input_path.text() or not self.dataset_base.text():
                raise ValueError("Input path and dataset base must be specified")
            
            if not self.folder_configs:
                raise ValueError("No COCO folders found")
            
            # Check label mappings
            label_mapping = self.mapping_table.get_mapping()
            if not label_mapping:
                raise ValueError("Label mapping cannot be empty")
            
            # Create config
            config = DatasetConfig(
                input_path=self.input_path.text(),
                dataset_base=self.dataset_base.text(),
                val_split_ratio=self.val_split.value(),
                test_split_ratio=self.test_split.value(),
                label_mapping=label_mapping,
                random_seed=self.random_seed.value(),
                create_yaml=self.create_yaml_toggle.isChecked(),
                folder_configs=self.folder_configs.copy()
            )
            
            self.convert_button.setEnabled(False)
            self.log_output.clear()
            self.log_message("Starting conversion...")

            # Create and start worker thread
            self.worker = ConversionWorker(config)
            self.worker.progress.connect(self.log_message)
            self.worker.error.connect(self.show_error)
            self.worker.finished.connect(self.conversion_finished)
            self.worker.start()
            
        except Exception as e:
            self.show_error(str(e))

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()