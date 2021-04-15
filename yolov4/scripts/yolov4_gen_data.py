import os

# Directory of project
root_dir = os.path.abspath("./")

# Directory of dataset
dataset_dir = os.path.join(root_dir, "dataset")

# Generate train.txt, test.txt, val.txt files
types = ['train','test','val']
for type in types:
    annotation_dir = os.path.join(dataset_dir, type+".json")
    image_dir = os.path.join(dataset_dir, type+"_images")

    image_dir_list = os.listdir(image_dir)

    f = open(os.path.join(dataset_dir, type+".txt"), "w")
    for image in image_dir_list:
        f.write((os.path.join(image_dir, image)+'\n').replace("\\",'/'))
    f.close()

# Generate masks.data file
f = open(os.path.join(dataset_dir, "masks.data"), "w")
f.write("classes = 2 \n"
        "train = " + os.path.join(dataset_dir, "train.txt").replace("\\",'/') + "\n"
        "valid = " + os.path.join(dataset_dir, "val.txt").replace("\\",'/') + "\n"
        "names = " + os.path.join(dataset_dir, "masks.names").replace("\\",'/') + "\n"
        "backup = " + os.path.join(dataset_dir, "backup").replace("\\",'/'))
f.close()

# Generate masks.names file
f = open(os.path.join(dataset_dir, "masks.names"), "w")
f.write("mask\nno_mask")
f.close()





