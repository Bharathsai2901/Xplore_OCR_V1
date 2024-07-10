import subprocess

def run_ocr():
    command = [
        "python3",
        "tools/infer/predict_system.py",
        "--image_dir=./test_imgs/2.jpg",
        "--det_model_dir=./inference/det/en_PP-OCRv3_det_infer/",
        "--cls_model_dir=./inference/cls/ch_ppocr_mobile_v2.0_cls_infer/",
        "--rec_model_dir=./inference/reg/en_PP-OCRv3_rec_infer/",
        "--rec_char_dict_path=./ppocr/utils/en_dict.txt"
    ]
    
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode == 0:
            print("Command executed successfully")
            print("Output:")
            print(result.stdout)
        else:
            print("Command failed with return code:", result.returncode)
            print("Error:")
            print(result.stderr)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_ocr()
