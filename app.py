import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QSpinBox,
    QDoubleSpinBox, QTextEdit, QGroupBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextBrowser, 
    QTabWidget, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import json
from dataclasses import dataclass
import shutil
import os
import markdown
import yaml
from typing import Dict, Optional, List, Tuple, Set
from sklearn.model_selection import train_test_split

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        # If the app is run as a bundled executable, _MEIPASS will exist.
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        # Otherwise, use the current working directory (or adjust as needed)
        base_path = Path(os.path.abspath("."))
    return base_path / relative_path


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

    def convert_annotations(self, json_file_path: Path) -> None:
        """Convert COCO JSON annotations to YOLO format"""
        try:
            with open(json_file_path) as f:
                data = json.load(f)

            category_mapping = {cat['id']: cat['name'] for cat in data['categories']}
            
            for index, image in enumerate(data['images']):
                image_id = image['id']
                img_width, img_height = image['width'], image['height']
                annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
                
                output_filename = (
                    f'{index}.txt' if self.config.rename_files 
                    else Path(image['file_name']).with_suffix('.txt').name
                )
                
                # Map original image name to the new one
                new_image_name = (
                    f"{index}{Path(image['file_name']).suffix}"
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

    def process_empty_annotations(self, images_folder: Path) -> None:
        """Create empty annotation files for images without annotations"""
        image_files = self.find_images(images_folder)
        
        if not image_files:
            raise ValueError(f"No image files found in {images_folder}")
            
        for index, image_file in enumerate(image_files):
            output_filename = (
                f'{index}.txt' if self.config.rename_files 
                else image_file.with_suffix('.txt').name
            )
            new_image_name = (
                f"{index}{image_file.suffix}"
                if self.config.rename_files
                else image_file.name
            )
            self.file_mapping[image_file.name] = new_image_name

            (self.label_output_dir / output_filename).touch()
            self.log(f"Created empty annotation file: {output_filename}")

    def copy_and_rename_images(self, source_folder: Path) -> None:
        """Copy and optionally rename images to the temporary output folder"""
        files = self.find_images(source_folder)
        
        if not files:
            raise ValueError(f"No image files found in {source_folder}")
            
        for index, file_path in enumerate(files):
            if self.config.rename_files:
                new_filename = f"{index}{file_path.suffix.lower()}"
            else:
                new_filename = file_path.name
                
            new_path = self.image_output_dir / new_filename
            shutil.copy2(file_path, new_path)

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
        # Find all CVAT COCO folders
        cvat_folders = self.find_cvat_coco_folders(self.config.input_path)
        
        if not cvat_folders:
            raise ValueError(f"No valid CVAT COCO folders found in {self.config.input_path}")
        
        self.log(f"Found {len(cvat_folders)} CVAT COCO folder(s):")
        for folder in cvat_folders:
            self.log(f"  - {folder.name}")

        # Process each folder
        for folder in cvat_folders:
            try:
                self.log(f"\nProcessing folder: {folder.name}")
                
                # Create a new config for this folder
                folder_config = DatasetConfig(
                    input_path=folder,
                    dataset_base=self.config.dataset_base,
                    val_split_ratio=self.config.val_split_ratio,
                    test_split_ratio=self.config.test_split_ratio,
                    label_mapping=self.config.label_mapping,
                    random_seed=self.config.random_seed,
                    rename_files=self.config.rename_files,
                    dataset_name=folder.name,
                    create_yaml=self.config.create_yaml
                )
                
                # Create a new converter instance for this folder
                folder_converter = COCOtoYOLOConverter(folder_config, self.progress_signal)
                folder_converter.process_single_folder(folder_config)
                
            except Exception as e:
                self.log(f"Error processing folder {folder.name}: {str(e)}")
                # Continue with next folder instead of stopping
                continue

    def process_single_folder(self, config: DatasetConfig) -> None:
        """Process a single CVAT COCO folder"""
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
                        
                    self.convert_annotations(json_files[0])
                else:
                    self.log(f"No JSON files found in {annotations_folder}, creating empty annotations")
                    self.process_empty_annotations(images_folder)
            else:
                self.log("No annotations folder found, creating empty annotations")
                self.process_empty_annotations(images_folder)

            # Copy and rename images
            self.copy_and_rename_images(images_folder)
            
            # Ensure we have files to organize
            if not os.path.exists(self.image_output_dir) or not os.listdir(self.image_output_dir):
                raise ValueError("No images were processed successfully")
            if not os.path.exists(self.label_output_dir) or not os.listdir(self.label_output_dir):
                raise ValueError("No labels were generated successfully")
                
            # 3) Split / organize
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

    def check_target_folders_exist(self, config: DatasetConfig) -> None:
        """
        Construct the final train/val(/test) subfolder paths for this dataset
        and raise an error if any already exist.
        """
        dataset_name = config.dataset_name
        train_image_dir = config.dataset_base / 'images' / 'train' / dataset_name
        val_image_dir   = config.dataset_base / 'images' / 'val' / dataset_name
        train_label_dir = config.dataset_base / 'labels' / 'train' / dataset_name
        val_label_dir   = config.dataset_base / 'labels' / 'val' / dataset_name

        dirs_to_check = [train_image_dir, val_image_dir, train_label_dir, val_label_dir]

        if config.test_split_ratio > 0:
            test_image_dir = config.dataset_base / 'images' / 'test' / dataset_name
            test_label_dir = config.dataset_base / 'labels' / 'test' / dataset_name
            dirs_to_check.append(test_image_dir)
            dirs_to_check.append(test_label_dir)

        # Identify any folders that exist
        already_existing = [str(d) for d in dirs_to_check if d.exists()]
        if already_existing:
            msg = (
                "Refusing to proceed because the following target folder(s) "
                "already exist for this dataset:\n"
                + "\n".join(f" - {p}" for p in already_existing)
            )
            raise FileExistsError(msg)


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
        self.setMinimumWidth(1000)
        self.setupUI()
        
    def setupUI(self):
        # Create a QTabWidget to hold multiple tabs (Conversion and Help)
        tab_widget = QTabWidget()
        self.setCentralWidget(tab_widget)
        
        # ----- Conversion Tab -----
        conversion_tab = QWidget()
        conversion_layout = QHBoxLayout(conversion_tab)
        
        # Left side container for groups (Path Config, Configuration, Log, Convert Button)
        left_container = QVBoxLayout()
        
        # Path configuration group
        path_group = QGroupBox("Path Configuration")
        path_layout = QVBoxLayout()
        # Input path
        input_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText(
            "Select COCO folder or parent containing COCO folders"
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
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        left_container.addWidget(self.log_output)
        
        # Convert button
        self.convert_button = QPushButton("Convert")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self.start_conversion)
        left_container.addWidget(self.convert_button)

        self.create_yaml_toggle = QCheckBox("Create/Update data.yaml")
        self.create_yaml_toggle.setChecked(True)
        left_container.addWidget(self.create_yaml_toggle)
        
        conversion_layout.addLayout(left_container, stretch=60)
        
        # Right side - Label Mapping (unchanged)
        mapping_group = QGroupBox("Label Mapping")
        mapping_layout = QVBoxLayout()
        self.mapping_table = LabelMappingTable()
        mapping_layout.addWidget(self.mapping_table)
        mapping_group.setLayout(mapping_layout)
        conversion_layout.addWidget(mapping_group, stretch=40)
        
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
            markdown_text = "# Help File Not Found\n\nPlease ensure 'help.md' exists in the correct location."

        # Convert markdown to HTML and set the content of the viewer
        try:
            html = markdown.markdown(markdown_text)
            self.markdown_viewer.setHtml(html)
        except Exception as e:
            # Fallback: display plain text if conversion fails
            self.markdown_viewer.setPlainText(markdown_text)

    def input_path_changed(self, new_path):
        """Handle input path changes and try to load JSON automatically"""
        if not new_path:  # Skip if path is empty
            return
            
        path = Path(new_path)
        if not path.exists():  # Skip if path doesn't exist
            return
            
        # Clear existing categories
        self.mapping_table.setRowCount(0)
        
        def process_cvat_folder(folder):
            json_files = list((folder / "annotations").glob("*.json"))
            if json_files:
                self.load_categories_from_file(json_files[0])
        
        # Check if the path is a CVAT folder itself
        if (path / "annotations").exists() and (path / "images").exists():
            process_cvat_folder(path)
        else:
            # Look for CVAT folders in immediate subdirectories only
            for item in path.iterdir():
                if (
                    item.is_dir() and 
                    (item / "annotations").exists() and 
                    (item / "images").exists()
                ):
                    process_cvat_folder(item)

    def update_convert_button_state(self):
        have_input_path = bool(self.input_path.text().strip())
        have_dataset_base = bool(self.dataset_base.text().strip())
        can_convert = have_input_path and have_dataset_base
        self.convert_button.setEnabled(can_convert)

    def load_categories_from_file(self, json_path):
        """Load categories from a specific JSON file and merge with existing ones"""
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            # Get existing categories
            existing_categories = set()
            for row in range(self.mapping_table.rowCount()):
                category_item = self.mapping_table.item(row, 0)
                if category_item:
                    existing_categories.add(category_item.text())
            
            # Add new categories
            new_categories = set(cat['name'] for cat in data['categories']) - existing_categories
            if new_categories:
                current_row_count = self.mapping_table.rowCount()
                self.mapping_table.setRowCount(current_row_count + len(new_categories))
                
                for i, category in enumerate(new_categories, start=current_row_count):
                    self.mapping_table.setItem(i, 0, QTableWidgetItem(category))
                    self.mapping_table.setItem(i, 1, QTableWidgetItem(str(i)))
                
                self.log_message(f"Added {len(new_categories)} new categories from {json_path.name}")
            
        except Exception as e:
            self.log_message(f"Failed to load categories from {json_path.name}: {str(e)}")

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit.setText(folder)

    def log_message(self, message):
        self.log_output.append(message)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.convert_button.setEnabled(True)

    def conversion_finished(self):
        self.log_message("Conversion completed successfully!")
        self.convert_button.setEnabled(True)

    def start_conversion(self):
        try:
            # Validate inputs
            if not self.input_path.text() or not self.dataset_base.text():
                raise ValueError("Input path and dataset base must be specified")

            label_mapping = self.mapping_table.get_mapping()
            if not label_mapping:
                raise ValueError("Label mapping cannot be empty")

            config = DatasetConfig(
                input_path=self.input_path.text(),
                dataset_base=self.dataset_base.text(),
                val_split_ratio=self.val_split.value(),
                test_split_ratio=self.test_split.value(),
                label_mapping=label_mapping,
                random_seed=self.random_seed.value(),
                create_yaml=self.create_yaml_toggle.isChecked()
            )

            self.log_message(f"Create YAML file: {'Yes' if self.create_yaml_toggle.isChecked() else 'No'}")

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

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit.setText(folder)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
