import os
import shutil
import xml.etree.ElementTree as ET
import random
import argparse
from tqdm import tqdm

def create_yolo_folders(base_path):
    """Create YOLO dataset folder structure"""
    os.makedirs(os.path.join(base_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'labels', 'test'), exist_ok=True)
    print(f"Created YOLO folder structure in {base_path}")

def convert_to_yolo_format(xml_file, image_width, image_height, classes):
    """Convert PASCAL VOC XML to YOLO format"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        yolo_lines = []
        
        for obj in root.findall('object'):
            # Get class name
            class_name_elem = obj.find('n')
            if class_name_elem is None or class_name_elem.text is None:
                # Try alternative tag names for class
                alternative_tags = ['name', 'class']
                found = False
                for tag in alternative_tags:
                    alt_elem = obj.find(tag)
                    if alt_elem is not None and alt_elem.text is not None:
                        class_name = alt_elem.text
                        found = True
                        break
                
                if not found:
                    print(f"Warning: Cannot find class name in object from {xml_file}")
                    object_text = ET.tostring(obj, encoding='unicode')
                    print(f"Object content: {object_text[:200]}...")  # Print first 200 chars to avoid overwhelming output
                    continue
            else:
                class_name = class_name_elem.text
            
            if class_name not in classes:
                print(f"Warning: Unknown class {class_name} in {xml_file}")
                continue
                
            class_id = classes.index(class_name)
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (center_x, center_y, width, height)
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin
            
            # Normalize coordinates
            x_center /= image_width
            y_center /= image_height
            width /= image_width
            height /= image_height
            
            # Format line as: class_id x_center y_center width height
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
        
        return yolo_lines
    except Exception as e:
        raise Exception(f"Error converting {xml_file}: {str(e)}")

def debug_xml_files(annotations_path, num_files=10):
    """Debug a sample of XML files"""
    print(f"Debugging {num_files} XML files...")
    
    xml_files = [f for f in os.listdir(annotations_path) if f.endswith('.xml')]
    sample_files = random.sample(xml_files, min(num_files, len(xml_files)))
    
    for xml_file in sample_files:
        xml_path = os.path.join(annotations_path, xml_file)
        print(f"\nAnalyzing {xml_file}:")
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Check size element
            size = root.find('size')
            if size is not None:
                width = size.find('width').text
                height = size.find('height').text
                print(f"  Size: {width}x{height}")
            else:
                print("  No size element found")
            
            # Check objects
            objects = root.findall('object')
            print(f"  Found {len(objects)} objects")
            
            for i, obj in enumerate(objects):
                # Try different class name tags
                class_name = None
                for tag in ['n', 'name', 'class']:
                    elem = obj.find(tag)
                    if elem is not None and elem.text is not None:
                        class_name = elem.text
                        tag_used = tag
                        break
                        
                bbox = obj.find('bndbox')
                if bbox is not None:
                    xmin = bbox.find('xmin').text
                    ymin = bbox.find('ymin').text
                    xmax = bbox.find('xmax').text
                    ymax = bbox.find('ymax').text
                    bbox_str = f"({xmin},{ymin}) to ({xmax},{ymax})"
                else:
                    bbox_str = "No bounding box found"
                
                print(f"  Object {i+1}: Class: {class_name} (tag: {tag_used if class_name else 'Not found'}), Bbox: {bbox_str}")
                
                # If class name not found, print the object for debugging
                if class_name is None:
                    print(f"  Object {i+1} content: {ET.tostring(obj, encoding='unicode')[:200]}...")
            
        except Exception as e:
            print(f"  Error analyzing {xml_file}: {str(e)}")

def process_files(files, split, annotations_path, images_path, output_path, classes, max_files=None):
    """Process files for a specific split (train/val/test)"""
    print(f"Processing {split} split...")
    
    success_count = 0
    error_count = 0
    
    # Limit the number of files if specified
    if max_files is not None:
        files = files[:max_files]
    
    for xml_file in tqdm(files):
        try:
            base_name = os.path.splitext(xml_file)[0]
            
            # Paths
            xml_path = os.path.join(annotations_path, xml_file)
            
            # Check for different possible image extensions
            img_src = None
            for ext in ['.png', '.jpg', '.jpeg']:
                test_path = os.path.join(images_path, f"{base_name}{ext}")
                if os.path.exists(test_path):
                    img_src = test_path
                    break
                    
            if img_src is None:
                print(f"Warning: Image for {base_name} not found. Skipping.")
                continue
                
            # Get file extension for destination
            _, ext = os.path.splitext(img_src)
            img_dst = os.path.join(output_path, 'images', split, f"{base_name}{ext}")
            txt_dst = os.path.join(output_path, 'labels', split, f"{base_name}.txt")
            
            # Parse XML and get image dimensions
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
                if size is not None:
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
                else:
                    # If size information is missing, try to get it from the image
                    from PIL import Image
                    with Image.open(img_src) as img:
                        width, height = img.size
                        print(f"Getting dimensions from image: {width}x{height}")
            except Exception as e:
                print(f"Error processing {xml_file}: {str(e)}")
                error_count += 1
                continue
            
            # Convert to YOLO format
            try:
                yolo_lines = convert_to_yolo_format(xml_path, width, height, classes)
            except Exception as e:
                print(f"Error converting {xml_file} to YOLO format: {str(e)}")
                error_count += 1
                continue
                
            if not yolo_lines:
                print(f"Warning: No valid annotations found in {xml_file}. Skipping.")
                continue
            
            # Copy image to destination
            shutil.copy(img_src, img_dst)
            
            # Write YOLO annotation
            with open(txt_dst, 'w') as f:
                f.write('\n'.join(yolo_lines))
                
            success_count += 1
                
        except Exception as e:
            print(f"Unexpected error processing {xml_file}: {str(e)}")
            error_count += 1
            continue
    
    print(f"Processed {split} split: {success_count} successful, {error_count} errors")
    return success_count, error_count

def prepare_dataset(dataset_path, output_path, val_split=0.15, test_split=0.15, max_files=None, debug=False):
    """Prepare the dataset by splitting it and converting annotations to YOLO format"""
    
    # The class mapping
    classes = ['helmet', 'head', 'person']  # Define classes in the correct order
    
    annotations_path = os.path.join(dataset_path, 'annotations')
    images_path = os.path.join(dataset_path, 'images')
    
    # Check if paths exist
    if not os.path.exists(annotations_path) or not os.path.exists(images_path):
        print(f"Error: annotations or images folder not found in {dataset_path}")
        return
    
    # Debug XML files if requested
    if debug:
        debug_xml_files(annotations_path, num_files=10)
        return
    
    # Create output directory structure
    create_yolo_folders(output_path)
    
    # Get all XML files
    xml_files = [f for f in os.listdir(annotations_path) if f.endswith('.xml')]
    random.shuffle(xml_files)  # Shuffle to ensure random split
    
    # Limit total files if specified
    if max_files is not None and max_files < len(xml_files):
        xml_files = xml_files[:max_files]
        print(f"Limited to {max_files} files for testing")
    
    # Calculate split indices
    val_size = int(len(xml_files) * val_split)
    test_size = int(len(xml_files) * test_split)
    train_size = len(xml_files) - val_size - test_size
    
    # Split the dataset
    train_files = xml_files[:train_size]
    val_files = xml_files[train_size:train_size + val_size]
    test_files = xml_files[train_size + val_size:]
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Process each split
    train_success, train_errors = process_files(train_files, 'train', annotations_path, images_path, output_path, classes)
    val_success, val_errors = process_files(val_files, 'val', annotations_path, images_path, output_path, classes)
    test_success, test_errors = process_files(test_files, 'test', annotations_path, images_path, output_path, classes)
    
    # Create data.yaml file
    create_data_yaml(output_path, classes, train_success, val_success, test_success)
    
    print("\nDataset preparation summary:")
    print(f"Train: {train_success} successful, {train_errors} errors")
    print(f"Val: {val_success} successful, {val_errors} errors")
    print(f"Test: {test_success} successful, {test_errors} errors")
    print(f"Total: {train_success + val_success + test_success} successful, {train_errors + val_errors + test_errors} errors")

def create_data_yaml(output_path, classes, train_size, val_size, test_size):
    """Create data.yaml configuration file for YOLO"""
    yaml_path = os.path.join(output_path, 'data.yaml')
    
    content = f"""# Dataset configuration for YOLOv8
path: {output_path}  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val  # Validation images (relative to 'path')
test: images/test  # Test images (relative to 'path')

# Classes
nc: {len(classes)}  # Number of classes
names: {str(classes).replace("'", '"')}  # Class names

# Statistics
train_size: {train_size}
val_size: {val_size}
test_size: {test_size}
"""
    
    with open(yaml_path, 'w') as f:
        f.write(content)
    
    print(f"Created data.yaml at {yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Helmet Detection dataset to YOLO format')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory for YOLO dataset')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process (for testing)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode to analyze XML files')
    
    args = parser.parse_args()
    
    prepare_dataset(args.dataset, args.output, args.val_split, args.test_split, args.max_files, args.debug)
