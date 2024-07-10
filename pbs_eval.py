import os
import cv2
import pandas as pd
import json

from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image

def create_pdf_with_reportlab(output_path, discrepancy_data):
    os.makedirs(output_path, exist_ok=True)
    # Create a canvas object with landscape orientation
    output_pdf_path = os.path.join(output_path, "ocr_results.pdf")
    text_file_path = os.path.join(output_path, "review_files.txt")
    c = canvas.Canvas(output_pdf_path, pagesize=landscape(letter))
    width, height = landscape(letter)

    with open(text_file_path, "w") as text_file:
        for data in discrepancy_data:
            filename, actual_plate, predicted_plate, discrepancy = data
            # Add image
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            display_width = width * 0.4
            display_height = display_width * aspect
            c.drawImage(ImageReader(img), width * 0.05, height * 0.5 - display_height / 2, width=display_width, height=display_height)

            #Add name
            c.setFont("Helvetica-Bold",15)
            text_name = width*0.2
            c.drawString(text_name, height * 0.8, f"{filename}")

            # Add text
            c.setFont("Helvetica", 12)
            text_x = width * 0.5 
            c.drawString(text_x, height * 0.65,f"Real Plate: {actual_plate}")
            c.drawString(text_x, height * 0.6, f"Predicted Plate: {predicted_plate}")
            c.drawString(text_x, height * 0.55, f"Discrepancy: {discrepancy}")

            # Write filename without extension to text file
            filename_without_extension = os.path.splitext(filename)[0]
            text_file.write(f"{filename_without_extension}\n")

            # Move to the next page
            c.showPage()

    # Save the PDF
    c.save()

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

def process_results(results_file, image_dir, output_path):
    with open(results_file, 'r') as file:
        lines = file.readlines()
    
    all_discrepancy_data = []
    discrepancy_counts = {'0D': 0, '1D': 0, '2D': 0, '3D': 0, '4D+': 0}
    count = 0
    # train_data = os.path.join(train_data_path,"train_data_cleaned.txt")
    # with open(train_data,'w') as train_file:
    for idx,line in enumerate(lines):
        line = line.strip()
        if(count%100==0):
            print(f'Processed {count} images')
        if line:
            # predicted_plate, actual_plate = line.split()
            parts = line.split()
            if len(parts) < 2:
                # Handle case where predicted plate is empty
                img_path = parts[0]
                predicted_plate = ""
            else:
                img_path, predicted_plate = parts
            image_name = os.path.basename(img_path)
            actual_plate = file_name_to_plate[image_name]
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                
                # Perform LCS and discrepancy analysis
                lcs_length, discrepancy = longest_common_subsequence(actual_plate, predicted_plate)
                
                # Update discrepancy counts
                if discrepancy == 0:
                    discrepancy_counts['0D'] += 1
                    # train_file.write(f"{image_name}\t{actual_plate}\n")
                elif discrepancy == 1:
                    discrepancy_counts['1D'] += 1
                elif discrepancy == 2:
                    discrepancy_counts['2D'] += 1
                elif discrepancy == 3:
                    discrepancy_counts['3D'] += 1
                else:
                    discrepancy_counts['4D+'] += 1
                
                # Generate PDF only if discrepancy is non-zero
                if discrepancy > 0:
                    all_discrepancy_data.append([image_name, actual_plate, predicted_plate, discrepancy])

            count = count+1
    print(f'Processed {count} images\n')
    # Calculate accuracy
    total_discrepancies = sum(discrepancy_counts.values())
    accuracy = discrepancy_counts['0D'] / total_discrepancies if total_discrepancies > 0 else 1

    create_pdf_with_reportlab(output_path,all_discrepancy_data)

    # Save discrepancy data to CSV
    os.makedirs(output_path, exist_ok=True)
    discrepancy_df = pd.DataFrame(all_discrepancy_data, columns=['Image Name', 'Actual Number', 'Detected Number', 'Discrepancy'])
    discrepancy_path = os.path.join(output_path, "discrepancies.csv")
    discrepancy_df.to_csv(discrepancy_path, index=False)

    print(f'PDF file saved to: {output_path}')
    print(f'Discrepancy details saved to: /Output_pbs/Test_GMDA/discrepancies.csv')
    print(f'Discrepancy counts: {discrepancy_counts}')
    print(f'Accuracy: {accuracy:.2%}')

def load_license_plate_map(map_file):
    with open(map_file, 'r') as file:
        file_name_to_plate = json.load(file)
    return file_name_to_plate


if __name__ == "__main__":
    results_file = 'license_plates.txt'
    image_dir = '/nfs/nas2VehiScan/IMPData/bharatp/Test_dump/images'
    map_file = '/nfs/nas2VehiScan/IMPData/bharatp/PaddleOCR/test_data/test_Bench/license_plate_map.json'
    output_path = '/nfs/nas2VehiScan/IMPData/bharatp/PaddleOCR/Output_pbs/Test_Bench_ModelV1/'
    # train_data_path = '/nfs/nas2VehiScan/IMPData/bharatp/PaddleOCR/train_data/data_cleaned'
    
    file_name_to_plate = load_license_plate_map(map_file)
    process_results(results_file, image_dir, output_path)
