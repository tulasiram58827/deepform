import json
import requests
import boto3
import os
import pandas as pd
import pytesseract
import numpy as np
import re
import cv2
import textract
import os
import tempfile
from pdf2image import convert_from_path
from PIL import Image
import glob

#rotate file if needed
def rotate(file, center = None, scale = 1.0):
    image = cv2.imread(file)
    page_rotation = int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    angle=360-page_rotation

    height, width = image.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to orig and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))

    return page_rotation, rotated
    
def convert_pdf(file_path, output_path):
    # save temp image files in temp dir, delete them after we are finished
    rotations = []

    # convert pdf to multiple image
    images = convert_from_path(file_path, output_folder='tmp/')
    
    # save images to temporary directory
    temp_images = []
    imlist = []
    for i in range(len(images)):
        image_path = f'tmp/{i}.jpg'
        images[i].save(image_path, 'JPEG')
        page_rotation, img = rotate(image_path)
        rotations.append(page_rotation)
        cv2.imwrite(image_path, img)
        imlist.append(Image.fromarray(img))
    
    imlist[0].save(output_path, compression="tiff_deflate", save_all=True,
                   append_images=imlist[1:])

    #clean up
    files = glob.glob('tmp/*')
    for f in files:
        os.remove(f)

    return output_path, rotations

pdf_api_error = []
manifest = pd.read_csv('fcc-updated-2020-sample.csv')
manifest=manifest.astype('object')

#setup the bucket
s3_key = os.environ['s3_key']
s3_key_secret = os.environ['s3_key_secret']
session = boto3.Session(aws_access_key_id=s3_key, aws_secret_access_key=s3_key_secret)
s3 = session.resource('s3')

for index, row in manifest.iterrows():
    #download pdf and save filename
    r_pdf = requests.get(row['FCC_URL'])
    filename = '{fileID}.pdf'.format(fileID=row['file_id'])

    #upload to S3 and record in csv 
    if r_pdf.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(r_pdf.content)

        OCR_needed = 'no' 
        OCR_error = ''
        page_rotation = ''
        #determine if file needs OCR and rotation
        words = textract.process(filename).split()

        if len(words) < 10:
            OCR_needed = 'yes' 
            try:
                clean_img, page_rotation = convert_pdf(filename, 'clean_img.tif')
                pdf = pytesseract.image_to_pdf_or_hocr(clean_img, extension='pdf')
                with open(filename, 'w+b') as f:
                    f.write(pdf) # pdf type is bytes by default
                os.remove(clean_img)
            except Exception as e: 
                OCR_error = e
        
        s3.meta.client.upload_file(Filename=filename, Bucket='fcc-updated-sample-2020', Key=filename)

        #remove temp pdf
        os.remove(filename)

        #update dataframe
        manifest.at[index, 'URL'] = 'https://fcc-updated-sample-2020.s3.us-east-2.amazonaws.com/{filename}'.format(filename = filename)
        manifest.at[index, 'OCR_needed'] = OCR_needed
        manifest.at[index, 'doc_angle'] = page_rotation
        manifest.at[index, 'OCR_error'] = OCR_error

    else:
        pdf_api_error.append(filename)

print(pdf_api_error)
manifest.to_csv('processed_training-sample.csv', index=False)
