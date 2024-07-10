import subprocess
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import multiprocessing

def run_ocr(image_path):
    command = [
        "python3",
        "tools/infer/predict_rec.py",
        f"--image_dir={image_path}",
        "--rec_model_dir=./inference/en_PP-OCRv3_rec/",
        "--rec_char_dict_path=./ppocr/utils/en_dict.txt"
    ]
    
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode == 0:
            print(f"OCR completed successfully for {image_path}")
            with open('license_plates.txt', 'r') as file:
                predictions = file.read().strip().split('\n')
            return predictions[0] if predictions else None
        else:
            print(f"OCR failed for {image_path} with return code:", result.returncode)
            print("Error:", result.stderr)
            return None
    except Exception as e:
        print(f"An error occurred during OCR for {image_path}: {e}")
        return None

def longest_common_subsequence(real, predicted):
    m, n = len(real), len(predicted)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if real[i - 1] == predicted[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    discrepancy = (len(real) - lcs_length) + max(0, len(predicted) - len(real))
    return lcs_length, discrepancy

def increase_contrast(image):
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

def preprocess_plate(plate):
    return plate.upper().replace(" ", "")

def process_image(image_data):
    folder_path, labels_path, filename = image_data
    discrepancy_data = []
    no_plate_data = []
    
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)
    contrasted_img = increase_contrast(img)
    contrasted_img_path = f'/tmp/contrasted_{filename}'
    cv2.imwrite(contrasted_img_path, contrasted_img)
    
    predicted_plate = run_ocr(contrasted_img_path)
    
    label_path = os.path.join(labels_path, os.path.splitext(filename)[0] + '.txt')
    with open(label_path, 'r') as file:
        real_plate = file.read().strip()
    
    if predicted_plate == "" or predicted_plate is None:
        no_plate_data.append([filename, real_plate, ''])
    else:
        lcs_length, discrepancy = longest_common_subsequence(real_plate, predicted_plate)
        discrepancy_data.append([filename, real_plate, predicted_plate, discrepancy])
    
    os.remove(contrasted_img_path)
    
    return discrepancy_data, no_plate_data

def process_images(folder_path, labels_path, output_pdf_path, no_plate_pdf_path):
    discrepancy_counts = {'0D': 0, '1D': 0, '2D': 0, '3D': 0, '4D+': 0}
    all_discrepancy_data = []
    all_no_plate_data = []

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_processes = multiprocessing.cpu_count()

    image_data = [(folder_path, labels_path, filename) for filename in image_files]
    
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(process_image, image_data)
    
    for discrepancy_data, no_plate_data in results:
        all_discrepancy_data.extend(discrepancy_data)
        all_no_plate_data.extend(no_plate_data)
    
    for _, _, _, discrepancy in all_discrepancy_data:
        if discrepancy == 0:
            discrepancy_counts['0D'] += 1
        elif discrepancy == 1:
            discrepancy_counts['1D'] += 1
        elif discrepancy == 2:
            discrepancy_counts['2D'] += 1
        elif discrepancy == 3:
            discrepancy_counts['3D'] += 1
        else:
            discrepancy_counts['4D+'] += 1

    discrepancy_df = pd.DataFrame(all_discrepancy_data, columns=['Image Name', 'Actual Number', 'Detected Number', 'Discrepancy'])
    no_plate_df = pd.DataFrame(all_no_plate_data, columns=['Image Name', 'Actual Number', 'Detected Number'])
    
    print(discrepancy_counts)
    discrepancy_df.to_csv('/Output_pbs/discrepancies.csv', index=False)
    no_plate_df.to_csv('/Output/no_plate_detected.csv', index=False)

    print(f'PDF file saved to: {output_pdf_path}')
    print(f'No plate detected PDF file saved to: {no_plate_pdf_path}')
    print(f'Discrepancy counts: {discrepancy_counts}')
    print(f'Discrepancy details saved to: /Output_pbs/discrepancies.csv')
    print(f'No plate detected details saved to: /Output/no_plate_detected.csv')

    generate_pdfs(all_discrepancy_data, all_no_plate_data, folder_path, output_pdf_path, no_plate_pdf_path)

def generate_pdfs(discrepancy_data, no_plate_data, folder_path, output_pdf_path, no_plate_pdf_path):
    with PdfPages(output_pdf_path) as pdf, PdfPages(no_plate_pdf_path) as no_plate_pdf:
        for data, pdf_file in [(discrepancy_data, pdf), (no_plate_data, no_plate_pdf)]:
            for item in data:
                filename, real_plate, predicted_plate = item[:3]
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                contrasted_img = increase_contrast(img)

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'OCR Results for {filename}', fontsize=16)

                ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax1.set_title('Original Image')
                ax1.axis('off')

                ax2.imshow(cv2.cvtColor(contrasted_img, cv2.COLOR_BGR2RGB))
                ax2.set_title('Contrast Enhanced Image')
                ax2.axis('off')

                ax3.axis('off')
                ax3.set_title('OCR Result')
                if predicted_plate:
                    discrepancy = item[3]
                    ax3.text(0.1, 0.9, predicted_plate, ha='left', va='top', fontsize=10, wrap=True)
                    ax3.text(0.1, 0.1, f'Real Plate: {real_plate}\nPredicted Plate: {predicted_plate}\nDiscrepancy: {discrepancy}', ha='left', va='top', fontsize=10, wrap=True)
                else:
                    ax3.text(0.5, 0.5, 'No number plate detected', ha='center', va='center', fontsize=12, color='red')

                plt.tight_layout()
                pdf_file.savefig(fig)
                plt.close()

if __name__ == "__main__":
    folder_path = '/nfs/nas2VehiScan/IMPData/bharatp/test_images'
    labels_path = '/nfs/nas2VehiScan/DataRepo/Training_Data/TrafficMon/ANPR/Data/LPRNet/TestBenchData/New/NewSingleRow/Test/labels'
    output_dir = '/nfs/nas2VehiScan/IMPData/bharatp/PaddleOCR/Output_pbs'
    output_pdf_path = os.path.join(output_dir, 'ocr_results.pdf')
    no_plate_pdf_path = os.path.join(output_dir, 'no_plate_detected.pdf')
    discrepancies_csv_path = os.path.join(output_dir, 'discrepancies.csv')
    no_plate_csv_path = os.path.join(output_dir, 'no_plate_detected.csv')
    
    process_images(folder_path, labels_path, output_pdf_path, no_plate_pdf_path)
