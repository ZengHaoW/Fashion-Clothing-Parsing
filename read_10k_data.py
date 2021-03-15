
# Hide the warning messages about CPU/GPU
import TensorflowUtils as utils
import glob
from tensorflow.python.platform import gfile
from six.moves import cPickle as pickle
import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'

"""
   get test and validation file list
   search the directory (*.jpg for input and *.png for annoation), save it into pickle file not to repeat the same serach process
"""


def read_dataset(data_dir):

    # sample record: {'image': f, 'annotation': annotation_file,
    # 'filename': filename}
    training_records = []

    traindir = "C:/Users/zx08x/Desktop/humanparsing/training/"

    print("## Training dir:", traindir)
    for filename in glob.glob(traindir + '*.jpg'):  # assuming jpg files
        record = {'image': filename, 'annotation': filename.replace(
            "images", "annotations").replace(
            "jpg", "png"), 'filename': filename}
        training_records.append(record)

    validation_records = []

    validationdir = "C:/Users/zx08x/Desktop/humanparsing/validation/"

    print("## Validation dir:", validationdir)
    for filename in glob.glob(
            validationdir + '*.jpg'):  # assuming jpg files
        record = {'image': filename,
                  'annotation': filename.replace("images", "annotations").replace("jpg", "png"),
                  'filename': filename}
        validation_records.append(record)

    logs_records = []

    logsdir = "test/"
    print("## logs dir:", logsdir)
    for filename in glob.glob(logsdir + '*.jpg'):  # assuming jpg files
        record = {'image': None, 'annotation': None, 'filename': None}
        record['image'] = filename
        record['filename'] = filename
        record['annotation'] = filename.replace(
            "jpg", "png")
        # record['label'] = filename.replace(
        #   "jpg", "png")
        logs_records.append(record)
    testing_records = []
    testdir = "C:/Users/zx08x/Desktop/humanparsing/test/"
    print("## Testing dir:", testdir)
    for filename in glob.glob(
            testdir + '*.jpg'):  # assuming jpg files
        record = {'image': None, 'annotation': None, 'filename': None}
        record['image'] = filename
        record['filename'] = filename
        record['annotation'] = filename.replace(
            "jpg", "png")
        record['label'] = filename.replace(
            "jpg", "png")
        testing_records.append(record)

    return training_records, validation_records, testing_records, logs_records


"""
    create image filename list for training and validation data (input and annotation)
    MIT SceneParsing data set dependent
"""


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]  # Linux
                # filename = os.path.splitext(f.split("\\")[-1])[0]  # windows
                annotation_file = os.path.join(
                    image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {
                        'image': f,
                        'annotation': annotation_file,
                        'filename': filename}
                    image_list[directory].append(record)
                else:
                    print(
                        "Annotation file not found for %s - Skipping: %s" %
                        (filename, annotation_file))

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print('No. of %s files: %d' % (directory, no_of_images))

    return image_list
