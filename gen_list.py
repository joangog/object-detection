import os

# Directory of project
root_dir = os.path.abspath("./")

# Directory of dataset
dataset_dir = os.path.join(root_dir, "dataset")
types = ['train','test','val']

for type in types:
    annotation_dir = os.path.join(dataset_dir, type+".json")
    image_dir = os.path.join(dataset_dir, type+"_images")

    image_dir_list = os.listdir(image_dir)

    f = open(os.path.join(dataset_dir, type+".txt"), "w")
    for image in image_dir_list:
        f.write((os.path.join(image_dir, image)+'\n').replace("\\",'/'))
    f.close()
