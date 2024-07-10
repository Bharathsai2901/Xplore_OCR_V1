import os
import shutil
import json

def create_training_set(image_dir, label_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_dir = os.path.join(output_dir, 'trainB')
    os.makedirs(train_dir, exist_ok=True)
    
    rec_gt_train_path = os.path.join(output_dir, 'trainB.txt')
    error_log_path = os.path.join(output_dir, 'error_log.txt')
    license_plate_map_path = os.path.join(output_dir, 'license_plate_mapB.json')

    license_plate_map = {}
    
    with open(rec_gt_train_path, 'w') as rec_gt_train_file, open(error_log_path, 'w') as error_log_file:
        count = 0
        
        for image_filename in os.listdir(image_dir):
            if image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_dir, image_filename)
                label_path = os.path.join(label_dir, os.path.splitext(image_filename)[0] + '.txt')
                
                if os.path.isfile(image_path) and os.path.isfile(label_path):
                    try:
                        with open(label_path, 'r') as label_file:
                            label = label_file.read().strip()
                        
                        new_image_path = os.path.join(train_dir, image_filename)
                        shutil.copy(image_path, new_image_path)
                        
                        rec_gt_train_file.write(f"{image_filename}\t{label}\n")
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
    image_dir = '/nfs/nas2VehiScan/Subhash/LPNetData/GMDA/'
    label_dir = '/nfs/nas2VehiScan/Subhash/LPNetData/GMDA/'
    output_dir = '/nfs/nas2VehiScan/IMPData/bharatp/PaddleOCR/train_data'  
    
    create_training_set(image_dir, label_dir, output_dir)
