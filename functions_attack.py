import cv2
import torch
from pathlib import Path
import time
import os
import numpy as np
import cvzone
import pandas as pd
import shutil
from IPython.display import Image, clear_output


# ANNOTATION CONVERSION

def yoloconversion(file, label, imagesize):
    dataframe = pd.read_csv(file, sep=" ", header=None)
    dataframe = dataframe.loc[dataframe[0] == label]

    xmin = (dataframe[1] - dataframe[3]/2) * imagesize[0]
    dataframe['xmin'] = xmin
    xmax = (dataframe[1] + dataframe[3]/2) * imagesize[0]
    dataframe['xmax'] = xmax
    ymin = (dataframe[2] - dataframe[4]/2) * imagesize[1]
    dataframe['ymin'] = ymin
    ymax = (dataframe[2] + dataframe[4]/2) * imagesize[1]
    dataframe['ymax'] = ymax
        
    if dataframe.shape[0] == 0:
        dataframe = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]
    else:
        dataframe = dataframe.to_numpy()
        dataframe.astype(int)

    return dataframe # [x, y, w, h, class, xmin, xmax, ymin, ymax]

# RUN THE MODEL

def run(model, file, label, img_dim): # (model, img file, label to analize, img_dimensions[height, width])
    #s = int(math.sqrt(img_dim[0] * img_dim[1])/2)
    #results = model(file, size=s)
    results = model(file)
    dataframe = results.pandas().xyxy[0]
    dataframe = dataframe.loc[dataframe['class'] == label]

    if dataframe.shape[0] == 0:
        dataframe = [[img_dim[1]/2, img_dim[0]/2, img_dim[1]/2, img_dim[0]/2, 0, label, '']]
    else:
        dataframe = dataframe.to_numpy()

    return dataframe # [xmin, ymin, xmax, ymax, confidence, class, class_name]

# INTERSECTION OVER UNION

def intersectionoverunion(b1, b2): # [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]
    dx = min(b1[2], b2[2]) - max(b1[0], b2[0])
    dy = min(b1[3], b2[3]) - max(b1[1], b2[1])

    try:
        iou = dx*dy / ((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - dx*dy)
    except ZeroDivisionError:
        iou = 0
    return iou

# CROP IMAGE

def cropimage(image, df_object): # df_object[class, x, y, w, h, xmin, xmax, ymin, ymax]
    xsx = int((df_object[6]-df_object[5])/2)
    xsy = int((df_object[8]-df_object[7])/2)

    xmin = int(df_object[5]) - xsx
    xmax = int(df_object[6]) + xsx
    ymin = int(df_object[7]) - xsy
    ymax = int(df_object[8]) + xsy

    bb_end = [xmax-xmin-xsx, ymax-ymin-xsy]

    if xmin < 0:
        xmin = 0
    if xmax > image.shape[1]:
        xmax = image.shape[1]
    if ymin < 0:
        ymin = 0
    if ymax > image.shape[0]:
        ymax = image.shape[0]

    return [xmin, ymin, xmax, ymax, xsx, xsy, bb_end[0], bb_end[1]] # [crop, bounding box]

##########################################################################################

def model_attack(dataset, model, model_size, iou_t, folders, color, labels, imagesize, grid):

    # Parameters

    input_path = 'traffic_signs/' + dataset + '/'
    path_output = 'traffic_signs/' + model + '_a/'

    model_path = 'traffic_signs/' + model + '/' + model_size
    model = torch.hub.load('', 'custom', path=model_path, source='local')
    model.iou = iou_t

    # Creating directories

    Path(path_output + 'patches/').mkdir(parents=True, exist_ok=True)
    Path(path_output + 'images/test/').mkdir(parents=True, exist_ok=True)
    Path(path_output + 'images/train/').mkdir(parents=True, exist_ok=True)
    Path(path_output + 'images/val/').mkdir(parents=True, exist_ok=True)
    Path(path_output + 'labels/test/').mkdir(parents=True, exist_ok=True)
    Path(path_output + 'labels/train/').mkdir(parents=True, exist_ok=True)
    Path(path_output + 'labels/val/').mkdir(parents=True, exist_ok=True)

    starttime = time.time()
    current = 0
    total = (len(os.listdir(input_path + 'images/test')) + len(os.listdir(input_path + 'images/train')) + len(os.listdir(input_path + 'images/test')))*len(color[0])

    for g in folders:
        images_path = input_path + 'images/' + g
        labels_path = input_path + 'labels/' + g
        images_path_a = path_output + 'images/' + g
        labels_path_a = path_output + 'labels/' + g
        files = os.listdir(images_path)
        files = [x.split('.')[0] for x in files]
        
        f = open(path_output + 'attack_output.txt', 'a')
        f.write('******************************\n')
        f.write('Directory: ' + g + '\n')
        f.write('******************************\n')
        f.close()
        for h in range(len(color[0])):
            
            opacy = []
            for i in files:
                img = cv2.imread(images_path + i + '.jpg')
                img_blank = cv2.imread(images_path + i + '.jpg')
                img_blank[:, :, :] = 0

                current = current + 1
                done = str(round((current/total)*100, 1))
                time_total = str(round(((time.time()-starttime)/60)*total/current, 1))
                time_left = str(round(((time.time()-starttime)*total/current - (time.time()-starttime))/60, 1))
                for j in labels:
                    df_obj = yoloconversion(labels_path+i+'.txt', j, imagesize) # [x, y, w, h, class, xmin, xmax, ymin, ymax]
                    df_run = run(model, images_path+i+'.jpg', j, imagesize) # [xmin, ymin, xmax, ymax, confidence, class, class_name]
                    for k in range(len(df_obj)): # k: object number
                        if sum(df_obj[k]) != sum([0, 0, 0, 0, 0, 0, 0, 0, 0]): # Is not an empty dataframe

                            iou_list = [] # List of iou in each object (in case the model detect several bounding boxes in one object)
                            for l in range(len(df_run)): # l: detected bounding box
                                iou_list.append(intersectionoverunion([df_obj[k][5], df_obj[k][7], df_obj[k][6], df_obj[k][8]], df_run[l]))
                                # [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]
                            iou = max(iou_list) # The intersection over union detected & object before atack

                            # Attack a cropped image
                            crop = cropimage(img, df_obj[k]) # [xmin, ymin, xmax, ymax, xsx, xsy, bb_end[0], bb_end[1]] # [crop, bounding box]
                            img_crop = img[crop[1]:crop[3], crop[0]:crop[2]]
                            df_run_crop = run(model, img_crop, j, img_crop.shape)
                            img_filter = np.zeros((img_crop.shape[0], img_crop.shape[1], img_crop.shape[2]+1)) # image height*width and 3+1 channels
                            for m in range(len(grid[0])): # m: patch sizes
                                grid_size = [] # [x, y]
                                grid_size.append((crop[6]-crop[4])/grid[0][m])
                                grid_size.append((crop[7]-crop[5])/grid[0][m])
                                for n in range(grid[0][m]): # n: coordinate x
                                    for o in range(grid[0][m]): # o: coordinate y
                                        start_point = (int(n*grid_size[0]) + crop[4], int(o*grid_size[1]) + crop[5]) # [x,y]
                                        end_point = (int((n+1)*grid_size[0]) + crop[4], int((o+1)*grid_size[1]) + crop[5]) # [x,y]
                                        img_crop_patch = img_crop.copy() # A cropped img with a single patch
                                        cv2.rectangle(img_crop_patch, start_point, end_point, color[0][h], -1) # (image, start_point, end_point, color, thickness)
                                        df_run_crop_patch = run(model, img_crop_patch, j, img_crop_patch.shape)
                                        iou_patch = intersectionoverunion([crop[4], crop[5], crop[6], crop[7]], df_run_crop_patch[0]) # [xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]
                                        if iou_patch < 0:
                                            iou_patch = 0
                                        elif iou_patch > iou:
                                            iou_patch = iou
                                        
                                        iou_dif = (iou - iou_patch)**1 *grid[1][m] # Attack intensity
                                        
                                        # Draw over the blank image
                                        img_patch = np.zeros((img_crop.shape[0], img_crop.shape[1], img_crop.shape[2]+1)) # transparent image [h, w, channels]
                                        img_patch[start_point[1]:end_point[1], start_point[0]:end_point[0], 3] = iou_dif # [ymin:ymax, xmin:xmax, channel] A patch img with a opacy of iou_difference value
                                        img_filter = img_filter + img_patch # Add the opacy with the factor multiplicador

                            # Calculate alpha
                            alpha = 0
                            if np.amax(img_filter) != 0: # Skip false positives
                                iou_alpha = 1
                                combo = 0 # veces seguidas que el ataque es sucessful
                                iterating = True
                                while iterating:
                                    alpha = alpha + 1
                                    img_filter_alpha = alpha/np.amax(img_filter) * img_filter
                                    img_filter_alpha[:, :, 0] = color[0][h][0] # set the Blue patch color
                                    img_filter_alpha[:, :, 1] = color[0][h][1] # set the Green patch color
                                    img_filter_alpha[:, :, 2] = color[0][h][2] # set the Red patch color
                                    cv2.imwrite(path_output + 'patches/' + i + color[1][h] + '_' + str(j) + '.png', img_filter_alpha) # save to load again
                                    img_filter_alpha = cv2.imread(path_output + 'patches/' + i + color[1][h] + '_' + str(j) + '.png', cv2.IMREAD_UNCHANGED) # Load the transparent patch unchanged
                                    img_crop_alpha = cvzone.overlayPNG(img_crop, img_filter_alpha) # image cropped with multiple patches
                                    df_run_crop_alpha = run(model, img_crop_alpha, j, img_crop_alpha.shape)
                                    iou_alpha = intersectionoverunion([crop[4], crop[5], crop[6], crop[7]], df_run_crop_alpha[0])
                                    if iou_alpha < 0.1:
                                        combo = combo + 1
                                    else:
                                        combo = 0
                                    if combo > 11:
                                        iterating = False
                                    if alpha > 254:
                                        iterating = False

                                opacy.append(alpha)
                                img_blank = cvzone.overlayPNG(img_blank, img_filter_alpha, pos=[crop[0], crop[1]])
                                img = cvzone.overlayPNG(img, img_filter_alpha, pos=[crop[0], crop[1]])
                
                cv2.imwrite(path_output + 'patches/' + i + color[1][h] + '.png', img_blank)
                cv2.imwrite(images_path_a + i + color[1][h] + '.jpg', img)
                shutil.copy(labels_path + i + '.txt', labels_path_a + i + color[1][h] + '.txt')
                
                f = open(path_output + 'attack_output.txt', 'a')
                f.write('Image: ' + i + color[1][h] + '.jpg' + ' --> Alpha: ' + str(round(alpha, 0)) + '\n')
                f.close()
                
                clear_output()
                print('Image: ' + i + color[1][h] + '.jpg' + ' --> Alpha: ' + str(round(alpha, 0)))
                print(done + '% --> ETA: ' + time_left + '/' + time_total + ' min')
    f = open(path_output + 'attack_output.txt', 'a')
    f.write('\nTotal images: ' + str(total) + '\n')
    f.write('Average alpha: ' + str(round(sum(opacy)/len(opacy), 0)) + '\n')
    f.write('Total time: ' + str(round((time.time() - starttime)/60, 1)) + ' min')
    f.close()
    clear_output()
    print('Average alpha: ' + str(round(sum(opacy)/len(opacy), 0)))
    print('Total: ' + str(round((time.time() - starttime)/60, 1)) + ' min')