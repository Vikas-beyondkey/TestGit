#This is script for Image Pre-Processing for common use.
#created by vikas | 14 April 2022 | 
#Version V-1.1 | 
# Tested on Python 3.7 | some of the code is not working on Python 3.8
import cv2
import os
import re
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import shutil

#Image Pre-Processing Prototype:
# * Read the image - Done
# * Write the image - Done
# * Converting from PDF to Image (with Poppler) - Done
# * Converting from PDF to Image (without Poppler)  (Need to work)
# * Operations:
# 	* Thresholding/ Binarization - Done
# 	* Adaptive Thresholding - Done
# 	* Noise Correction  - Done
# 	* Resize the image by width & height - Done
# 	* Resize the image by scale - Done
# 	* Background Removal (white background) - Done
# 	* Background Removal (white background with all side margin) - Done
# 	* Background Removal (Other background/ Crop Form from other background) [Image Regstration] - Done
# -----------------------------------------
#   * Background Removal (Other background/ Crop Form from other background) (In-progress)
# 	* Skew Correction (In-progress - Testing)
#   * Converting from PDF to Image (without Poppler)  (In-progress)
#   * Vertical Line remove
#   * Horizontal Line remove
#   * Convert images into PDF
#   * 

# For next version
# * OCR by pytessract



#Read the image file from the image path
def ReadImage(_imgpath):
    '''
    Read the image from the imagepath.
    _imgpath - Complete image path for read the imagefile.
    '''
    img = ''
    try:
        img = cv2.imread(_imgpath)
    except Exception as ee:
        print(ee)
    return img

#Write the image file on the image path
def WriteImage(_img, _imgpath):
    '''
    Write the image on the imagepath.
    _img - image which you want to write.
    _imgpath - Complete image path where you want to write image.
    '''
    IsWrite = False
    try:
        cv2.imwrite(_imgpath,_img)
        IsWrite = True 
    except Exception as ee:
        print(ee)
    return IsWrite

#Show the image 
def ShowImage(_title, _img):
    '''
    Show the image with the title.
    _title - title of the window, which is shown on the window.
    _img - image which you want to show.
    '''
    try:
        cv2.imshow(_title, _img)
        cv2.waitKey(0)
    except Exception as ee:
        print(ee)

#Here we resize the image by scale percentage.
def ResizeImage(img, scale_percent = 60):
    '''
    Here we resize the image by scale percentage.
    img - image ndArray to process the image.
    scale_percent - how much size is required in percentage (default = 60).
    '''
    resized = img.copy()
    try:
        # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    except Exception as ee:
        print(ee)
    return resized

#Here we resize the image by Width & Height. 
def ResizeImageBydimension(img, width = 745, height = 745):
    '''
    Here we resize the image by fixied Size.
    img - image ndArray to process the image.
    width - required image width size (default width = 745).
    height - required image height size (default height = 745).
    '''
    resized = img.copy()
    try:
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) #good results
    except Exception as ee:
        print(ee)
    return resized

#Here we check and create a folder.
def CheckAndCreateFolder(_folderName_Path):
    """
    Here it will check the folder is exists or not, if not then it will create it.
    _folderName_Path - folder path which you want to check or create the folder.
    """
    IsFolderCrearte =False
    try:
        if not os.path.exists(_folderName_Path):
            os.makedirs(_folderName_Path)
            print('New Diroctory is created',_folderName_Path)
            IsFolderCrearte = True
        return IsFolderCrearte   
    except Exception as e:
        print(e)

def RemoveDirectory(dirpath):
    """
    Here delete the directory.
    dirpath - folder path which you want to delete.
    """
    IsDirDelete =False
    try:
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
            print('Diroctory Removed ',dirpath)
            IsDirDelete = True
        return IsDirDelete   
    except Exception as e:
        print(e)

#Here we Convert Pdf to Image 
def ConvertPdfToImage(_filename, working_dir, NoofPages=-1, Quality=0):
    '''
    Here we convert the pdf to image.
    filename - File which you want to convert from pdf to jpg.
    working_dir - Working directory where it will store the output image.
    NoofPages - here we want only first page then pass 1 & all then pass -1. also specify 2,3,4 pages.
    '''
    try:
        tempDirectory=working_dir+"\\"+'TempImages\\'+os.path.splitext(os.path.basename(_filename))[0]
        #check if this temp file is not exists then create dir.    
        CheckAndCreateFolder(tempDirectory)
        if Quality==1: #High Quality
            images_from_path = convert_from_path(_filename, output_folder=tempDirectory, dpi=300, jpegopt={"quality": 100, "progressive": True,    "optimize": True})
        else: #Normal Quality
            images_from_path = convert_from_path(_filename, output_folder=tempDirectory)
        
        base_filename  =  os.path.splitext(os.path.basename(_filename))[0] + '.jpg'     
        save_dir =os.path.join(working_dir, 'PdfToImage')
        RemoveDirectory(save_dir)
        CheckAndCreateFolder(save_dir)    

        filePath=[]
        _counter = 0
        for i in range(0,len(images_from_path)):  
            base_filename  =  os.path.splitext(os.path.basename(_filename))[0] +'_'+ str(_counter) + '.jpg' 
            fileName=os.path.join(save_dir, base_filename)

            #Save the converted file in the following loaction.    
            images_from_path[i].save(fileName, 'JPEG')

            filePath.append(fileName)
            _counter = _counter + 1
            if NoofPages!=-1:
                if NoofPages==_counter:
                    break  #Here we need only first page of the file.
        return filePath, tempDirectory 
    except Exception as e:
        print(e)

#This is used for creating the PDF from the images.
def Create_img2pdf(Imgfolderpath, _outFolder, _filename):
    '''
    Here we create a pdf file from images.
    Imgfolderpath - Path of the folder where all the images are stored. from which you want to create PDF file.
    _outFolder - Folder path where you want to store the PDF file.
    _filename - Name of the PDF file.
    '''
    IsPDFCreated = False
    imagelist = []
    outfile = ''
    try:
        listfiles = os.listdir(Imgfolderpath)
        for index, imgname in enumerate(listfiles):
            ext = imgname.split('.')[1]
            if ext.lower()!='pdf':
                imgpath = os.path.join(Imgfolderpath, imgname)
                if index==0:
                    img1 = Image.open(imgpath)
                    im1 = img1.convert('RGB')
                else:
                    img = Image.open(imgpath)
                    im = img.convert('RGB')
                    imagelist.append(im)
        outfile = os.path.join(_outFolder,_filename)
        im1.save(outfile,save_all=True, append_images=imagelist)
        IsPDFCreated = True
        print("Pdf genrated. please check.")
    except Exception as ee:
        print(ee)
    return outfile, IsPDFCreated

#====================== Image Pre-Processing ===============================
'''
This is used for clean the image from some noises.
_image - use ndarray of image.
thresh - Threshold value which is used to  
'''
def Image_threshold(_image, thresh = 127):
    outimage = _image.copy()
    try:
        # convert the image to grayscale and flip the foreground
        # and background to ensure foreground is now "white" and
        # the background is "black"
        if len(_image.shape)==2:
            gray = outimage 
        else:
            gray = cv2.cvtColor(outimage, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.bitwise_not(gray) #invert the image.
        # threshold the image, setting all foreground pixels to 255 and all background pixels to 0
        outimage = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        outimage = cv2.bitwise_not(outimage) #invert the image.
    except Exception as ee:
        print(ee)
    return outimage

#Noise Removal - This is used to clear the image from salt & pepper noise.
def NoiseRemoval(_image, _kernel = 3, Filter_size = 5):
    '''
    This is used to clear the image from salt & pepper noise.
    _image - img ndarray 
    _kernel - kernel used to remove the small dots type noise. (default = 3)
    Filter_size - Large filters are very slow, so it is recommended to use d=5 for real-time applications, and perhaps d=9 for offline applications that need heavy noise filtering.
    '''
    out_bil = ''
    try:
        # load the input image and convert it to grayscale
        image = _image.copy()
        out_img = Image_threshold(image)
        out_bil= cv2.medianBlur(out_img, _kernel) #work well in salt & peper noise
        out_bil = cv2.bilateralFilter(out_bil,Filter_size,6,6) #to make the edges sharp
    except Exception as ee:
        print(ee)
    return out_bil

#Here we remove the white background. if error occure it will return orignal image.
def RemoveWhiteBackground(_img, rightMargin = 10):
    '''
    Here we remove the white background from the image.
    _img - Here we pass the image as ndarray
    rightMargin - right side margin.
    '''
    outimg = _img.copy()
    try:
        #gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        gray = NoiseRemoval(_img, 5, 5)  #clean some noise from the image.
        gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
        coords = cv2.findNonZero(gray) # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        outimg = _img[y:y+h, x:x+w+rightMargin] # Crop the image - note we do this on the original image
    except Exception as ee:
        print(ee)
    return outimg

#Here we remove the white background. if error occure it will return orignal image.
def RemoveWhiteBackground_WithMargin(_img, Margin = 10):
    '''
    Here we remove the white background from the image.
    _img - Here we pass the image as ndarray
    Margin - add the margin in all the 4 sides.
    '''
    outimg = _img.copy()
    try:
        #update by vikas on 29 March 2022 | Remove some text from image.
        #trim the bottom if the text in bottom will remove.
        # bottom_margin = 80 #Remove the part from the image. (some text in the bottom in some images)
        # left_margin = 5 #This will remove the 5 pixel from image.(some line here)
        # height, width = _img.shape[0:2]
        # _img = _img[0:height-bottom_margin, 0+left_margin:width]

        #gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        top_margin = Margin    
        right_margin = Margin
        left_margin = Margin
        bottom_margin = Margin 

        gray = NoiseRemoval(_img, 5, 5)  #clean some noise from the image.
        gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
        coords = cv2.findNonZero(gray) # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        
        if y < top_margin + 1:
            top_margin = 0 #Here we reset the value to avoid getting error.
        if x < left_margin + 1:
            left_margin = 0 #Here we reset the value to avoid getting error.

        outimg = _img[y-top_margin:y+h+bottom_margin, x-left_margin:x+w+right_margin] # Crop the image - note we do this on the original image
    except Exception as ee:
        print(ee)
    return outimg

#here we apply the adeptive threshold to mange the lighting efect in thresholds.
def ApplyAdeptiveThreshold(_img):
    '''
    Adeptive Threshold is make the image gray with controling the lighting effect in image.
    _img - pass the image in ndarray.
    '''
    outimage = _img
    try:
        if len(_img.shape)==2:
            _inputimage = outimage 
        else:
            _inputimage = cv2.cvtColor(outimage, cv2.COLOR_BGR2GRAY)
        blocksize = 25
        constant = 8
        max_value = 255 # 8 bits
        outimage = cv2.adaptiveThreshold(
            src=_inputimage, 
            maxValue=max_value ,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
            thresholdType=cv2.THRESH_BINARY, 
            blockSize=blocksize , 
            C=constant,
        )
    except Exception as ee:
        print(ee)
    return outimage

#Here Image Registration is used for cropping the forms from a image which is having background.
def Image_Registration(img_sample, img_traget, ORB_feature = 5000):
    '''
    Here Image Registration is used for cropping the forms from a image which is having background.
    img_sample - Sample Image of the form, like form which you want in the target form.
    img_traget - Here we pass the target Image which you want to crop form.
    '''
    transformed_img = img_traget
    try:
        
        # Convert to grayscale.
        img1 = cv2.cvtColor(img_traget, cv2.COLOR_BGR2GRAY)
        
        img2 = cv2.cvtColor(img_sample, cv2.COLOR_BGR2GRAY)
        height, width = img2.shape
        #print(img1.shape)
        
        # img1 = ApplyAdeptiveThreshold(img1) 
        # img1 = NoiseRemoval(img1, 5, 5)  #clean some noise from the image.
        # img2 = ApplyAdeptiveThreshold(img2)
        #cv2.imwrite(outputfile_T, img1)
        # Create ORB detector with 5000 features.
        orb_detector = cv2.ORB_create(ORB_feature)

        # Find keypoints and descriptors.
        # The first arg is the image, second arg is the mask
        # (which is not reqiured in this case).
        kp1, d1 = orb_detector.detectAndCompute(img1, None)
        kp2, d2 = orb_detector.detectAndCompute(img2, None)

        # Match features between the two images.
        # We create a Brute Force matcher with
        # Hamming distance as measurement mode.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        # # match the features
        # method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        # matcher = cv2.DescriptorMatcher_create(method)

        # Match the two sets of descriptors.
        matches = matcher.match(d1, d2)

        # # Need to draw only good matches, so create a mask
        # matchesMask = [[0,0] for i in range(len(matches))]

        # Sort matches on the basis of their Hamming distance.
        matches.sort(key = lambda x: x.distance)  #======= This part of code is working on Python 3.7 | It throw error on Python 3.8

        # # Draw first 10 matches.
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30], flags=2, outImg=None)
        # img3 = ResizeImage(img3,20)
        # ShowImage("result", img3)

        # Take the top 90 % matches forward.
        matches = matches[:int(len(matches)*90)]
        no_of_matches = len(matches)
        
        # Define empty matrices of shape no_of_matches * 2.
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = kp1[matches[i].queryIdx].pt
            p2[i, :] = kp2[matches[i].trainIdx].pt
        
        # Find the homography matrix.
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

        # Use this matrix to transform the
        # colored image wrt the reference image.
        transformed_img = cv2.warpPerspective(img_traget, homography, (width, height))

    except Exception as ee:
        print(ee) 
    return transformed_img 

#Here we correct the skew images.
def Skew_Correction(img):
    '''
    Here we correct the skew images.
    img - just pass the image it will corrected it and return back.
    '''
    result = img.copy()
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Invert the image, Its give better results 
        gray_inv = cv2.bitwise_not(gray)

        thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cords = np.column_stack(np.where (thresh>0))

        #here we find the angle of the higest Y (start point) and endpoint Y. 
        angle = cv2.minAreaRect(cords)[-1]

        (h,w) = img.shape[:2] #the the height & width of the image
        centre = ( w//2 , h//2) #Get the center of the image to rotate the image.
        if angle < -45:
            angle = -(90 + angle)
        elif angle >= 45:
            angle = 90 - angle    
        else:
            angle = -angle

        #Here we rotate the image by the given angle with center points
        matrix_rotate = cv2.getRotationMatrix2D(centre, angle, 1.0) 
        result = cv2.warpAffine(result, matrix_rotate, (w,h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)

    except Exception as ee:
        print(ee)
    return result

#here we remove the vertical & horizontal line.
def RemoveVertical_Horizontal_Lines(_img, _ksize = 12):
    image = _img.copy()
    try:
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _ksize))
        detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_ksize, 1))
        detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))

        result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
        
    except Exception as ex:
        print(ex)
    return result