import os
import shutil
import cv2
import json

inc_contrast = False

def increase_contrast(imagepath):
    image = cv2.imread(imagepath)
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrasted_gray = clahe.apply(gray)

    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = contrasted_gray
        lab = cv2.merge((l, a, b))
        contrasted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        contrasted = contrasted_gray

    return contrasted

def create_training_set(image_dir, label_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_dir = os.path.join(output_dir, 'test_14k')
    contrasted_test_dir = os.path.join(test_dir, 'contrasted')
    
    os.makedirs(test_dir, exist_ok=True)
    if inc_contrast:
        os.makedirs(contrasted_test_dir, exist_ok=True)
    
    rec_gt_test_path = os.path.join(output_dir, 'test_14k.txt')
    error_log_path = os.path.join(output_dir, 'error_log.txt')
    license_plate_map_path = os.path.join(output_dir, 'license_plate_map.json')
    
    license_plate_map = {}

    with open(rec_gt_test_path, 'w') as rec_gt_test_file, open(error_log_path, 'w') as error_log_file:
        count = 0
        
        for image_filename in os.listdir(image_dir):
            if image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_dir, image_filename)
                label_path = os.path.join(label_dir, os.path.splitext(image_filename)[0] + '.txt')
                
                if os.path.isfile(image_path) and os.path.isfile(label_path):
                    try:
                        with open(label_path, 'r') as label_file:
                            label = label_file.read().strip()
                        
                        new_image_path = os.path.join(test_dir, image_filename)
                        shutil.copy(image_path, new_image_path)
                        
                        if inc_contrast:
                            contrasted_img = increase_contrast(image_path)
                            contrasted_img_path = os.path.join(contrasted_test_dir, image_filename)
                            cv2.imwrite(contrasted_img_path, contrasted_img)
                            
                        rec_gt_test_file.write(f"{image_filename}\t{label}\n")
                        license_plate_map[label] = image_filename
                        count += 1
                        
                        if count % 1000 == 0:
                            print(f"Processed {count} images")
                    
                    except Exception as e:
                        error_log_file.write(f"Failed to process {image_filename}: {e}\n")
                else:
                    error_log_file.write(f"Missing file: {image_filename}\n")
    
    with open(license_plate_map_path, 'w') as f:
        json.dump(license_plate_map, f, indent=4)
    
    print(f"Total images processed: {count}")
    print(f"License plate map saved to {license_plate_map_path}")

if __name__ == "__main__":
    image_dir = '/nfs/nas2VehiScan/IMPData/bharatp/Test_dump/images'
    label_dir = '/nfs/nas2VehiScan/IMPData/bharatp/Test_dump/labels'
    output_dir = '/nfs/nas2VehiScan/IMPData/bharatp/PaddleOCR/test_data'  
    
    create_training_set(image_dir, label_dir, output_dir)
