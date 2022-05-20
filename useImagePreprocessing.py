import os
import Image_Pre_Processing as imgpre
import cv2



if __name__=="__main__":
    #filepath = 'E:/ImagePre_Preocessing/SkewImages/test1.jpg'
    filepath = 'E:/ImagePre_Preocessing/TestImages/16565488_1_0.jpg'
    img = img1 = imgpre.ReadImage(filepath)
    
    #========Write image in the file system.========
    #imgpre.WriteImage(img,'E:/ImagePre_Preocessing/TestImages/Testwrite.jpg')

    # #========Remove white Background with rightside margin ========
    # img = imgpre.RemoveWhiteBackground(img,10)
    
    # #========Resize the image with the scale percentage.========
    # img = imgpre.ResizeImage(img,40)

    # #========Resize image by dimensions========
    # img = imgpre.ResizeImageBydimension(img,width= 700, height= 500)
    # imgpre.ShowImage("Test", img)
    

    # #========Remove  white background with all side margin.========
    # img1 = imgpre.RemoveWhiteBackground_WithMargin(img1,10)
    # img1 = imgpre.ResizeImage(img1,40)
    # imgpre.ShowImage("Test1", img1)

    # #======== Convert PDF to image ========
    # #pdfFile = 'E:/ImagePre_Preocessing/PDF_test/multipages.pdf'
    # pdfFile = 'E:/Leads/Langtech/PDFs/Gettysburg (Delta Dental)_07.pdf'
    # WORKING_DIR = 'E:/ImagePre_Preocessing/WORKING_DIR'
    # lstimages, _temp = imgpre.ConvertPdfToImage(pdfFile,WORKING_DIR,-1) #with poppler
    # print(lstimages)
    # imgpre.RemoveDirectory(_temp)
    # filespath = os.path.join(WORKING_DIR,'PdfToImage')
    # _outfilepath, IsPdfCreated = imgpre.Create_img2pdf(filespath,WORKING_DIR,'Test.pdf')
    # print(_outfilepath)

    # #========Image Alignment [Crop the background of the form]========
    # imgSample = imgpre.ReadImage('E:/ImagePre_Preocessing/TestImages/Blue_withBackground/SampleImage/bluesample.jpg')
    # imgAlign = imgpre.Image_Registration(imgSample, img )
    # imgAlign = imgpre.ResizeImage(imgAlign,40)
    # imgpre.ShowImage("Image Align",imgAlign)
    
    # # ========Image Registration (for crop the form from background images)========
    # imgSample = imgpre.ReadImage('E:/ImagePre_Preocessing/TestImages/yellow_withBackground/Sample/yellowSample.jpg')
    
    # dirpath = 'E:/ImagePre_Preocessing/TestImages/yellow_withBackground' #/Test
    # lstfiles = os.listdir(dirpath)
    # for imgpath in lstfiles:
    #     imgFullpath = os.path.join(dirpath,imgpath)
    #     if os.path.isfile(imgFullpath):
    #         imgTest = imgpre.ReadImage(imgFullpath)
    #         imgAlign = imgpre.Image_Registration(imgSample, imgTest, ORB_feature= 5000 )
    #         imgAlign = imgpre.ResizeImage(imgAlign,40)
    #         imgpre.ShowImage("Image Align",imgAlign)

    # #========Skew correction============
    filepath = 'E:/ImagePre_Preocessing/SkewImages/Skewed_Images/Screenshot 2022-05-16 205614.jpg' #Screenshot 2022-05-16 210016
    img = cv2.imread(filepath)
    newimage = imgpre.Skew_Correction(img)
    #newimage = imgpre.ResizeImage(newimage,40)

    cv2.imshow("orignal", img)
    cv2.imshow("output", newimage)
    cv2.waitKey(0)

    #Extract Form without image registration.
    