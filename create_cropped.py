import cv2
import pandas as pd
import os


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

def create_crop(in_path, out_path, directories, imagesize, label_list):

    os.mkdir(out_path)
    os.mkdir(out_path + 'labels/')
    os.mkdir(out_path + 'images/')

    for k in directories:
        files = os.listdir(in_path + 'images/' + k)
        files = [x.split('.')[0] for x in files]

        os.mkdir(out_path + 'labels/' + k)
        os.mkdir(out_path + 'images/' + k)

        for i in files:
            file_img = in_path + 'images/' + k + i + '.jpg'
            file_label = in_path + 'labels/' + k + i + '.txt'

            img = cv2.imread(file_img)
            df_obj = yoloconversion(file_label, label_list[0], imagesize)[0] # only one object in the img

            w = df_obj[6] - df_obj[5]
            h = df_obj[8] - df_obj[7]

            x2 = df_obj[5] - w/2
            y2 = df_obj[7] - h/2
            x3 = df_obj[6] + w/2
            y3 = df_obj[8] + h/2

            if x3 > imagesize[0]:
                delta_x = x3 - imagesize[0]
            elif x2 < 0:
                delta_x = x2
            else:
                delta_x = 0

            if y3 > imagesize[1]:
                delta_y = y3 - imagesize[1]
            elif y2 < 0:
                delta_y = y2
            else:
                delta_y = 0

            if delta_x < 0:
                x_new = (w + delta_x)/(2*w + delta_x)
            else:
                x_new = 1 - (w - delta_x)/(2*w - delta_x)

            if delta_y < 0:
                y_new = (h + delta_y)/(2*h + delta_y)
            else:
                y_new = 1 - (h - delta_y)/(2*h - delta_y)

            w_new = w/(2*w - abs(delta_x))
            h_new = h/(2*h - abs(delta_y))

            f = open(out_path + 'labels/' + k + i + '_crop.txt', 'a')
            f.write('0 ' + str(x_new) + ' ' + str(y_new) + ' ' + str(w_new) + ' ' + str(h_new))
            f.close()

            if x2 < 0:
                x2 = 0
            if x3 > imagesize[0]:
                x3 = imagesize[0]
            if y2 < 0:
                y2 = 0
            if y3 > imagesize[1]:
                y3 = imagesize[1]

            df_obj_extra = [int(x2), int(y2), int(x3), int(y3)]
            img_crop = img[df_obj_extra[1]:df_obj_extra[3], df_obj_extra[0]:df_obj_extra[2]]
            cv2.imwrite(out_path + 'images/' + k + i + '_crop.jpg', img_crop)