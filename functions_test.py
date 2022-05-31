import pandas as pd
import numpy as np
import time
from IPython.display import Image, clear_output
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import os

# YOLO ANNOTATION CONVERSION

def yoloconversion(txtfile, label, imagesize):
    dataframe = pd.read_csv(txtfile, sep=" ", header=None)
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

    return dataframe # [class, x, y, w, h, xmin, xmax, ymin, ymax]

# RUN THE MODEL

def run(model, img_file, label, img_dim): # model, img file, label to analize, img dimensions(height, width)
    s = int(min(img_dim[0], img_dim[1])/2)
    results = model(img_file, size=s)
    dataframe = results.pandas().xyxy[0]
    dataframe = dataframe.loc[dataframe['class'] == label]

    if dataframe.shape[0] == 0:
        dataframe = [[img_dim[1]/2, img_dim[0]/2, img_dim[1]/2, img_dim[0]/2, 0, label, '']]
    else:
        dataframe = dataframe.to_numpy()

    return dataframe # [xmin, ymin, xmax, ymax, confidence, class, class_name]

# INTERSECTION OVER UNION

def intersectionoverunion(b1, b2):
    dx = min(b1[2], b2[2]) - max(b1[0], b2[0])
    dy = min(b1[3], b2[3]) - max(b1[1], b2[1])
    
    try:
        iou = dx*dy / ((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - dx*dy)
    except ZeroDivisionError:
        iou = 0
    return iou

# BOUNDING BOXES vs ANNOTATIONS MATRIX

def bbannotation(annotation_df, model_df):
    annotations_bb = []
    for k in range(int(len(annotation_df))):
        annotations_bb.append([annotation_df[k][5], annotation_df[k][7], annotation_df[k][6], annotation_df[k][8]])

    model_bb = []
    for k in range(int(len(model_df))):
        model_bb.append([model_df[k][0], model_df[k][1], model_df[k][2], model_df[k][3]])

    matrix = []
    for j in range(int(len(model_df))):
        matrix2 = []
        for k in range(int(len(annotation_df))):
            matrix2.append(intersectionoverunion(model_bb[j], annotations_bb[k]))
        matrix.append(matrix2)

    return matrix

# CONFIDENCE IOU MATRIX

def bbiou(matrix, model_df): # added model_df
    matriz = np.array(matrix)

    iou_values = list(range(len(matrix)))
    for i in range(len(iou_values)):
        maximum = np.amax(matriz)
        location = np.where(matriz == maximum)
        iou_values[location[0][0]] = maximum
        matriz[location[0][0], :] = -10
        matriz[:, location[1][0]] =  -10
    
    confidenceiou = []
    for k in range(int(len(model_df))):
        confidenceiou.append(model_df[k][4])

    return [confidenceiou, iou_values]

# PRECISION RECALL TABLE

def table(conf_iou, iou, objects): # added iou, objects
    b3 = []
    b4 = []
    b5 = []
    b6 = []
    acctp = 0
    accfp = 0

    for i in conf_iou[1]:
        if i < iou:
            b3.append(0)
            b4.append(1)
            accfp = accfp + 1
            b5.append(acctp)
            b6.append(accfp)
        else:
            b3.append(1)
            b4.append(0)
            acctp = acctp + 1
            b5.append(acctp)
            b6.append(accfp)
    conf_iou = np.vstack((conf_iou, b3))
    conf_iou = np.vstack((conf_iou, b4))
    conf_iou = np.vstack((conf_iou, b5))
    conf_iou = np.vstack((conf_iou, b6))

    # PRECISION AND RECALL
    b7 = []
    b8 = []
    for i in range(len(conf_iou[0])):
        b7.append(conf_iou[4][i] / (conf_iou[4][i] + conf_iou[5][i]))
        b8.append(conf_iou[4][i] / objects)
    conf_iou = np.vstack((conf_iou, b7))
    conf_iou = np.vstack((conf_iou, b8))

    return conf_iou

# PLOT

def plot_map(tabla):
    x = tabla[7]
    x = np.insert(x, 0, 0)
    y = tabla[6]
    y = np.insert(y, 0, y[0])
    coordenadas = [x, y]

    # AREA
    #area = 0
    #for i in range(len(x) - 1):
    #    area = area + ((x[i+1] - x[i]) * (y[i+1] + y[i])/2)

    return(coordenadas)


# MAP several IOU

def test_map_several(model, label_list, imagesize, test_dataset, model_pytorch):

    output_dir = 'traffic_signs/' + test_dataset + '/test_results/' + model + '/'
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except:
        pass

    path_images = 'traffic_signs/' + test_dataset + '/images/'
    path_labels = 'traffic_signs/' + test_dataset + '/labels/'

    if len(os.listdir(path_images)) == 3: # If is not a testing directory
        path_images = path_images + 'test/'
        path_labels = path_labels + 'test/'
        
    images = os.listdir(path_images)
    files = [x.split('.')[0] for x in images]

    model_path = 'traffic_signs/' + model + '/' + model_pytorch
    model = torch.hub.load('', 'custom', path=model_path, source='local')

    model.iou = 0.05 # starting value

    IOU_values = [0.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
    map_values = []

    f = open(output_dir + 'IOU_vs_mAP.csv', 'a')
    f.write('IOU,mAP\n')
    starttime = time.time()
    done = 0
    total = len(IOU_values)*len(label_list)

    for o in range(len(IOU_values)): # o = iou
        iou = IOU_values[o]

        objects = 0 # Real objects in the image
        conf_iou_matrix =[[], []] # correlate real bounding boxes vs detected bounding boxes
        axis = []
        areas = []

        for l in label_list: # l = label
            mAP = []

            for i in range(len(files)): # i = number of images [0, 1, 2, ...]
                txtfile = path_labels + files[i] + '.txt'
                imgfile = path_images + files[i] + '.jpg'

                img = cv2.imread(imgfile)
                annotation_df = yoloconversion(txtfile, l, imagesize)
                model_df = run(model, imgfile, l, imagesize)
                matrix = bbannotation(annotation_df, model_df) # BOUNDING BOXES vs ANNOTATIONS MATRIX
                conf_iou = bbiou(matrix, model_df) # CONFIDENCE IOU MATRIX

                for j in conf_iou[0]:
                    conf_iou_matrix[0].append(j)

                for k in conf_iou[1]:
                    conf_iou_matrix[1].append(k)

                if annotation_df[0][1] + annotation_df[0][2] != 0:
                    objects = len(annotation_df) + objects

            arr2D = -np.array(conf_iou_matrix)
            sortedArr = -arr2D [ :, arr2D[0].argsort()]

            tabla = table(sortedArr, iou, objects)
            axis_label = plot_map(tabla)
            axis.append(axis_label)

            done = done + 1
            tiempo = time.time()-starttime
            tiempo_total = tiempo*total/done
            clear_output()
            print(str(round(done/total*100, 1)) + '% --> ETA: ' + str(round((tiempo_total-tiempo)/60, 1)) + '/' + str(round((tiempo_total)/60, 1)) + ' min')

        for m in label_list:
            x = axis[m][0]
            y = axis[m][1]

            area = 0
            for n in range(len(x) - 1):
                area = area + ((x[n+1] - x[n]) * (y[n+1] + y[n])/2)
            areas.append(area)

        f.write(str(IOU_values[o]) + ',' + str(sum(areas)/len(areas)) + '\n')
        map_values.append(sum(areas)/len(areas))

    f.write('Total time,' + str(round((time.time()-starttime)/60, 1)) + ' min\n')
    f.write('Date,' + str(time.ctime(int(time.time()))))
    f.close()

    clear_output()
    print('Total time: ' + str(round((time.time()-starttime)/60, 1)) + ' min')
    print('mAP 0.05-0.95 @0.05: ' + str(sum(map_values)/len(map_values)))
    plt.plot(IOU_values, map_values)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.savefig(output_dir + 'iou_map.png')
    plt.clf()
    Image(filename=output_dir + 'iou_map.png', width=1200)


# MAP IOU threshold

def test_map_single(iou_t, model, label_list, imagesize, test_dataset, model_pytorch):

    output_dir = 'traffic_signs/' + test_dataset + '/test_results/' + model + '/'
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except:
        pass

    path_images = 'traffic_signs/' + test_dataset + '/images/'
    path_labels = 'traffic_signs/' + test_dataset + '/labels/'

    if len(os.listdir(path_images)) == 3: # If is not a testing directory
        path_images = path_images + 'test/'
        path_labels = path_labels + 'test/'
        
    images = os.listdir(path_images)
    files = [x.split('.')[0] for x in images]

    model_path = 'traffic_signs/' + model + '/' + model_pytorch
    model = torch.hub.load('', 'custom', path=model_path, source='local')

    objects = 0
    conf_iou_matrix =[[], []]
    axis = []
    areas = []

    current = 0
    total = len(label_list)

    for l in label_list:
        label = label_list[l]
        mAP = []

        for i in range(len(files)):
            txtfile = path_labels + files[i] + '.txt'
            imgfile = path_images + files[i] + '.jpg'

            img = cv2.imread(imgfile)
            annotation_df = yoloconversion(txtfile, label, imagesize)
            model_df = run(model, imgfile, label, img.shape)
            matrix = bbannotation(annotation_df, model_df)
            conf_iou = bbiou(matrix, model_df)

            for j in conf_iou[0]:
                conf_iou_matrix[0].append(j)

            for k in conf_iou[1]:
                conf_iou_matrix[1].append(k)

            if annotation_df[0][1] + annotation_df[0][2] != 0:
                objects = len(annotation_df) + objects

        current = current + 1
        clear_output()
        print('Done: ' + str(round(current/total*100, 1)))

        arr2D = -np.array(conf_iou_matrix)
        sortedArr = -arr2D [ :, arr2D[0].argsort()]

        tabla = table(sortedArr, iou_t, objects)
        axis_label = plot_map(tabla)
        axis.append(axis_label)

    for m in label_list:
        x = axis[m][0]
        y = axis[m][1]
        plt.plot(x, y)

        area = 0
        for n in range(len(x) - 1):
            area = area + ((x[n+1] - x[n]) * (y[n+1] + y[n])/2)
        areas.append(area)

        f = open(output_dir + 'PRcurve_' + str(m) + '.csv', 'a')
        f.write('Recall,Precision\n')
        for o in range(len(x)):
            f.write(str(x[o]) + ',' + str(y[o]) + '\n')
        f.close()

    clear_output()
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.savefig(output_dir+'PRcurve.png')
    plt.clf()
    print(areas)
    print("mAP: " + str(sum(areas)/len(areas)))
    Image(filename=output_dir+'PRcurve.png', width=1200)