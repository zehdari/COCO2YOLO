# COCO2YOLO Help

---

## How to Use

### 1. Setting Up Paths

**Input Path:**  

- Click the **Browse...** button to select a folder that contains your COCO dataset(s). The app will scan for subdirectories with the proper `images` and `annotations` folders.

**Dataset Base:**  

- Specify the base directory where the converted dataset will be saved. This directory will eventually contain subdirectories such as `images/train`, `images/val`, and `labels/train`, etc.

### 2. Configuring Conversion Parameters

**Label Mapping:**  

- The application auto-loads categories from the first detected JSON file. You can manually adjust the YOLO IDs in the mapping table.

**Split Ratios:**  

- **Validation Split Ratio:** Set the fraction of images to reserve for validation (e.g., `0.2` for 20%).

- **Test Split Ratio:** Optionally set aside a portion of the dataset for testing.

**Random Seed:**  

- Choose a random seed to ensure reproducible splits.

### 3. Starting the Conversion

Click the **Convert** button.  

**Logging:** The log window will display real-time messages showing progress, including folder detection, file copying, and split statistics.

**Completion:** When finished, the dataset is organized into the appropriate directories, and a `data.yaml` file is generated or updated in the dataset base directory.

---

## Troubleshooting

**No Valid Folders Found:**  

- Ensure the selected input path contains subfolders with both `images` and `annotations` directories.
- Verify that your JSON annotation files are correctly formatted.

**Missing Label Mappings:**  

- If new categories appear in your annotations that aren’t mapped, update the label mapping table manually.
- Check the log messages for any warnings about missing mappings.

**File Count Mismatch:**

- The conversion process expects a matching number of image and annotation files. If there’s a mismatch, check the integrity of your input dataset.

**Target Directory Exists Error:**  

- The app will not overwrite existing target folders. Remove or rename existing directories if needed.

**Insufficient Samples for Split:**  

- If there are not enough images for the chosen split ratios, reduce the split ratios or increase your dataset size.

---

## FAQ

**Q: What annotation format is supported?**  
*A:* Only COCO (CVAT) JSON annotations are supported.

**Q: Can I modify YOLO IDs after conversion?**  
*A:* Yes, you can adjust the mapping in the label mapping table before starting the conversion.

**Q: What happens if target directories already exist?**  
*A:* The application will raise an error to avoid overwriting existing data. Please ensure the dataset base is clean or back up the existing directories.

**Q: I’m having trouble with file splits. What should I do?**  
*A:* Verify that your dataset contains enough images for the selected split ratios, and check the log for any specific error messages.
