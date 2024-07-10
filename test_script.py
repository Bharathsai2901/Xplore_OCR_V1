import subprocess
import os

def run_ocr(image_path):
    command1 = [
        "python3",
        "tools/infer/predict_system.py",
        f"--image_dir={image_path}",
        "--det_model_dir=./inference/det/en_PP-OCRv3_det_infer/",
        "--cls_model_dir=./inference/cls/ch_ppocr_mobile_v2.0_cls_infer/",
        "--rec_model_dir=./inference/reg/en_PP-OCRv3_rec_infer/",
        "--rec_char_dict_path=./ppocr/utils/en_dict.txt"
    ]

    command2 = [
        "python3",
        "tools/infer/predict_rec.py",
        f"--image_dir={image_path}",
        "--rec_model_dir=./inference/en_PP-OCRv3_rec/",
        "--rec_char_dict_path=./ppocr/utils/en_dict.txt"
    ]
    
    try:
        result = subprocess.run(command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode == 0:
            print(f"Command executed successfully for {image_path}")
            print("Output:")
            print(result.stdout)
        else:
            print(f"Command failed for {image_path} with return code:", result.returncode)
            print("Error:")
            print(result.stderr)
    except Exception as e:
        print(f"An error occurred for {image_path}: {e}")

def process_images(image_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, filename)
            run_ocr(image_path)

if __name__ == "__main__":
    # Replace 'path_to_images' with the directory containing your 1000 images
    # process_images('path_to_images')
    run_ocr('/nfs/nas2VehiScan/IMPData/bharatp/Test1/DS0500_2019111500068_nonHSRP_Single_1_lp.jpg')


