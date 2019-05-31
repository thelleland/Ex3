import sys
import getopt
import shutil
import os
import random
import skimage as sk
from scipy import ndarray
from skimage import transform
from skimage import util
from math import floor



def get_args():
    size = None
    folder = None
    
    argv = sys.argv[1:]
    
    try:
        opts, args = getopt.getopt(argv, "n:c:")
    except getopt.GetoptError as err:
        print(err)
        opts = []

    for opt, arg in opts:
        if opt in ['-n']:
            size = arg
        elif opt in ['-c']:
            folder = arg


    return size, folder



    #horizontal flip
def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]

def vertical_flip(image_array: ndarray):
    return image_array[::-1, :]

available_transformations = {
    'horizontal_flip': horizontal_flip,
    'vertical_flip' : vertical_flip
}




#Counting files in class folders, code form website, ref no.1
def fileCount(folder):
    "count the number of files in a directory"

    count = 0

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if os.path.isfile(path):
            count += 1
        elif os.path.isfolder(path):
            count += fileCount(path)    

    return count


def augment_folder(size, folder):
    
    # random num of transformations to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))
    
    num_transformatinos = 0
    transformation_image = None
    
    # loop on all files of the folder and build a list of files paths
    images = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    

    num_files = fileCount(folder)
    num_generated_files = 0
    
    while num_files < size:
        # random image from the folder
        image_path = random.choice(images)
        
        # read image as an two dimensional array of pixels
        image_to_transform = sk.io.imread(image_path)
        
        # random num of transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformation_image = None
        
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1
        
        # define a name for new file
        new_file_path = '%s/augmented_image_%s.jpg' % (folder, num_generated_files)
        
        #write image to the disk
        sk.io.imsave(new_file_path, transformed_image)
        num_files += 1
        num_generated_files += 1




    

def main():
    size, folder = get_args()

    augment_folder(int(size), folder)




if __name__ == "__main__":
    main()
