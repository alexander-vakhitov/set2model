import urllib2  # the lib that handles the url stuff
import os, shutil

def download_flowers(flower_list, write_to_folder):
    cnt = 0
    for line in open(flower_list, 'r'):
        class_name = line[0:-1]
        class_folder = write_to_folder+'/'+class_name
        if (os.path.exists(class_folder)):
            continue
        os.makedirs(class_folder)
        download_images_for_query(class_name, class_folder)
        cnt += 1
