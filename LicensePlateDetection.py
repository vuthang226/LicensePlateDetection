# remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp, load_model, preprocess_image
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob



wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)


# Load model architecture, weight and labels
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")

def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

# user wpod to detect plate
def get_plate(image_path, Dmax=608, Dmin = 256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, lp_type, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg,lp_type, cor

##



def sort_contours(cnts,plateHeight,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    lines = [(0) if b[1]/plateHeight <= 0.3 else (1) for b in boundingBoxes]
    
    (cnts, boundingBoxes,lines) = zip(*sorted(zip(cnts, boundingBoxes, lines),
                                        key=lambda b: (b[2],[b[1][i]]), reverse=reverse))
    return cnts




def handelPlate(LpImg,lp_type,type,numLv,gaussianLv = 13):
    #type 1 blur + otsu 2 gray + otsu 3 gray + lv 4 gaussian
     #check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        
        plate_image = cv2.convertScaleAbs(LpImg, alpha=(255.0))
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        if(type == 1 or type == 4):
            blur = cv2.GaussianBlur(gray,(numLv,numLv),0)
            binary = cv2.threshold(blur, 180, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            if(type == 4):
                binary = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV,gaussianLv,3)
        else:
            binary = cv2.threshold(gray, 180, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            if(type == 3):
                binary = cv2.threshold(gray, numLv, 255,
                            cv2.THRESH_BINARY_INV )[1]
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _  = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # creat a copy version "test_roi" of plat_image to draw bounding box
        test_roi = plate_image.copy()

        # Initialize a list which will be used to append charater image
        crop_characters = []

        # define standard width and height of character
        digit_w, digit_h = 30, 60
        ratioForPlate = [0.575,0.65]
        for c in sort_contours(cont,plate_image.shape[0]):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1<=ratio<=3.60: # Only select contour with defined ratio
                if h/(plate_image.shape[0]/lp_type)>=ratioForPlate[lp_type-1]: # Select contour which has the height larger than 50% of the plate
                    # Draw bounding box arroung digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                    # Sperate number and gibe prediction
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)
        return test_roi,crop_characters



#code call func

while True:
    test_image_path = input("Path image:")
    if(test_image_path == "quit"):
        print("End")
        break
    vehicle, LpImg,lp_type,cor = get_plate(test_image_path)
    for index in range(len(LpImg)):
        parameters = [(7,0,-2),(1,2,1),(75,121,5),(7,0,-2)]
        flag = False
        maxSeg = 0
        crop_charactersLast = []
        for i in range(1,5):
            if flag == True: break 
            for j in range(parameters[i-1][0],parameters[i-1][1],parameters[i-1][2]):
                if flag == True: break 
                if(i ==4):
                    for k in range(13,2,-2):
                        if flag == True: break 
                        test_roi,crop_characters= handelPlate(LpImg[index],lp_type[index],i,j,k)
                        if maxSeg < len(crop_characters):
                            maxSeg = len(crop_characters)
                            crop_charactersLast = crop_characters
                        if len(crop_characters) >=8: 
                            crop_charactersLast = crop_characters
                            flag = True
                else:
                    test_roi,crop_characters= handelPlate(LpImg[index],lp_type[index],i,j)
                    if maxSeg < len(crop_characters):
                        maxSeg = len(crop_characters)
                        crop_charactersLast = crop_characters
                    if len(crop_characters) >=8: 
                        crop_charactersLast = crop_characters
                        flag = True
            
        fig = plt.figure(figsize=(14,4))
        grid = gridspec.GridSpec(ncols=len(crop_charactersLast),nrows=1,figure=fig)

        for i in range(len(crop_charactersLast)):
            fig.add_subplot(grid[i])
            plt.axis(False)
            plt.imshow(crop_charactersLast[i],cmap="gray")
        plt.show()   


        fig1 = plt.figure(figsize=(15,3))
        cols = len(crop_charactersLast)
        grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig1)

        final_string = ''
        for i,character in enumerate(crop_charactersLast):
            fig1.add_subplot(grid[i])
            title = np.array2string(predict_from_model(character,model,labels))
            plt.title('{}'.format(title.strip("'[]"),fontsize=20))
            final_string+=title.strip("'[]")
            plt.axis(False)
            plt.imshow(character,cmap='gray')
        plt.show()
        #Chuoi ki tự bien so cuoi cung
        print(final_string)


    
