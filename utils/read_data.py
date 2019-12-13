import glob
import csv
import numpy as np # linear algebra

def parst_data_labels(file):
    csv_file = open(file,'r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    data_labels = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            recording = {
                "filename"           : row[0],
                "isliestatment"     : row[1]
            }
            data_labels.append(recording)
            line_count += 1
    print(f'Processed {line_count} lines.')
    return data_labels

def read_labeld_image_list(datafolder, fileender , lable_file) :
    labels_raw = parst_data_labels(lable_file)
    audio_list = []
    label_list = []
    for entry in labels_raw:
        audio_list.append(datafolder + entry["filename"])
        if entry["isliestatment"] == "True":    # [is lie , is true]
            label_list.append([0,1])
        else:
            label_list.append([1,0])
    return np.array(audio_list) , np.array(label_list)

