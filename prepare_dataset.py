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
            # Get class name - try different tags used in XML files
            class_name = None
            for tag in ['name', 'n', 'class']:
                class_elem = obj.find(tag)
                if class_elem is not None and class_elem.text is not None:
                    class_name = class_elem.text
                    break
                    
            if class_name is None:
                print(f"Warning: Could not find class name in {xml_file}, object will be skipped")
                continue
                
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
        raise Exception(f"Error in {xml_file}: {str(e)}")

def prepare_dataset(dataset_path, output_path, val_split=0.15, test_split=0.15):
    """Prepare the dataset by splitting it and converting annotations to YOLO format"""
    
    # The class mapping
    classes = ['helmet', 'head', 'person']  # Define classes in the correct order
    
    annotations_path = os.path.join(dataset_path, 'annotations')
    images_path = os.path.join(dataset_path, 'images')
    
    # Check if paths exist
    if not os.path.exists(annotations_path) or not os.path.exists(images_path):
        print(f"Error: annotations or images folder not found in {dataset_path}")
        return
    
    # Create output directory structure
    create_yolo_folders(output_path)
    
    # Get all XML files
    xml_files = [f for f in os.listdir(annotations_path) if f.endswith('.xml')]
    random.shuffle(xml_files)  # Shuffle to ensure random split
    
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
    process_files(train_files, 'train', annotations_path, images_path, output_path, classes)
    process_files(val_files, 'val', annotations_path, images_path, output_path, classes)
    process_files(test_files, 'test', annotations_path, images_path, output_path, classes)
    
    # Create data.yaml file
    create_data_yaml(output_path, classes, train_size, val_size, test_size)
    
    print("Dataset preparation complete!")

def process_files(files, split, annotations_path, images_path, output_path, classes):
    """Process files for a specific split (train/val/test)"""
    print(f"Processing {split} split...")
    
    success_count = 0
    error_count = 0
    
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
    
    args = parser.parse_args()
    
    prepare_dataset(args.dataset, args.output, args.val_split, args.test_split)
