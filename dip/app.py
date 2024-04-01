
from flask import Flask, request, render_template
import numpy as np
import pydicom as pdcm
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import cv2
from PIL import Image
import os

app = Flask(__name__)

def clear_uploads():
    upload_folder = app.config['UPLOAD_FOLDER']
    file_extensions = ['jpg', 'jpeg', 'png']  
    for filename in os.listdir(upload_folder):
        if any(filename.lower().endswith(ext) for ext in file_extensions):
            file_path = os.path.join(upload_folder, filename)
            os.remove(file_path)

def normalization(img):
  mean = np.mean(img)
  std = np.std(img)
  img = img-mean
  img = img/std
  return img

def intensity_reduction(img):
  img = normalization(img)
  middle = img[100:400,100:400]
  mean = np.mean(middle)
  max = np.max(img)
  min = np.min(img)

  img[img==max]=mean
  img[img==min]=mean
  return img, middle

def thresholding(img):
  img, middle = intensity_reduction(img)
  img = median_filter(img,size=3)
  img = anisotropic_diffusion(img)
  kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
  centers = sorted(kmeans.cluster_centers_.flatten())
  threshold = np.mean(centers)
  thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
  return thresh_img

def mask_generation(img):
  img = thresholding(img)
  eroded = morphology.erosion(img,np.ones([4,4]))
  dilation = morphology.dilation(eroded,np.ones([10,10]))
  labels = measure.label(dilation)
  label_vals = np.unique(labels)
  regions = measure.regionprops(labels)
  good_labels = []

  for prop in regions:
      B = prop.bbox
      if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
          good_labels.append(prop.label)
  mask = np.ndarray([512,512],dtype=np.int8)
  mask[:] = 0


  for N in good_labels:
      mask = mask + np.where(labels==N,1,0)
  mask = morphology.dilation(mask,np.ones([10,10]))
  return mask

def final_image(img):
    mask = mask_generation(img)
    return mask*img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'select' in request.form:
            if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'selected_image.jpg')):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'selected_image.jpg'))
            return render_template('select.html')
        
        if 'original' in request.form:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_image.jpg')
            img = pdcm.dcmread(img_path).pixel_array
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_image.jpg')
            plt.imshow(img)
            plt.savefig(img_path,format='jpg')
            plt.close()
            return render_template('main.html', message='Original', image_path=img_path)
        
        if 'normalize' in request.form:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_image.jpg')
            img = pdcm.dcmread(img_path).pixel_array
            normalized_img = normalization(img)
            normalized_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'normalized.jpg')
            plt.imshow(normalized_img)
            plt.savefig(normalized_img_path,format='jpg')
            plt.close()
            return render_template('main.html', message='Image normalized', image_path=normalized_img_path)
        
        if 'intensity' in request.form:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_image.jpg')
            img = pdcm.dcmread(img_path).pixel_array
            intensity_red, _ = intensity_reduction(img)
            intensity_red_path = os.path.join(app.config['UPLOAD_FOLDER'], 'intensity_reduced.jpg')
            plt.imshow(intensity_red)
            plt.savefig(intensity_red_path,format='jpg')
            plt.close()
            return render_template('main.html', message='Intensity Reduced', image_path=intensity_red_path)
        
        if 'threshold' in request.form:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_image.jpg')
            img = pdcm.dcmread(img_path).pixel_array
            threshold_img = thresholding(img)
            threshold_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'thresholded.jpg')
            plt.imshow(threshold_img)
            plt.savefig(threshold_img_path,format='jpg')
            plt.close()
            return render_template('main.html', message='After Thresholding', image_path=threshold_img_path)
        
        if 'mask' in request.form:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_image.jpg')
            img = pdcm.dcmread(img_path).pixel_array
            mask_img = mask_generation(img)
            mask_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask.jpg')
            plt.imshow(mask_img)
            plt.savefig(mask_img_path,format='jpg')
            plt.close()
            return render_template('main.html', message='Image Mask', image_path=mask_img_path)
        
        if 'final' in request.form:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_image.jpg')
            img = pdcm.dcmread(img_path).pixel_array
            final_img = final_image(img)
            final_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'final.jpg')
            plt.imshow(final_img)
            plt.savefig(final_img_path,format='jpg')
            plt.close()
            return render_template('main.html', message='Final Segmentation', image_path=final_img_path)
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('capture.html', message='No file uploaded')

    file = request.files['file']

    if file.filename == '':
        return render_template('capture.html', message='No file selected')

    if file:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_image.jpg')
        file.save(img_path)

        # Disable the capture button
        return render_template('main.html', message='Image uploaded successfully', image_path=img_path,
                               capture_disabled=True)

    return render_template('main.html', message='Image upload failed')


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'static'
    app.run(host='0.0.0.0', port=5000)