import os
import h5py
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def load_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    train_path = os.path.join(base_dir, "datasets/trainset.hdf5")
    test_path = os.path.join(base_dir, "datasets/testset.hdf5")

    train_dataset = h5py.File(train_path, "r")
    X_train = np.array(train_dataset["X_train"][:])
    y_train = np.array(train_dataset["Y_train"][:])

    test_dataset = h5py.File(test_path, "r")
    X_test = np.array(test_dataset["X_test"][:])
    y_test = np.array(test_dataset["Y_test"][:])

    return X_train, y_train, X_test, y_test


def load_cat_and_dog_data(image_size=(150, 150), test_size=0.2, random_state=42 , max_images_per_class=2000):

    # Make sure to download the dataset from https://www.microsoft.com/en-us/download/details.aspx?id=54765 and place it in the datasets folder before running this function.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    cat_dir = os.path.join(base_dir, "datasets", "cat")
    dog_dir = os.path.join(base_dir, "datasets", "dog")

    images = []
    labels = []

    # ------------------------
    # Load cat images (label = 0)
    # ------------------------
    print("Loading cat images...")
    for filename in os.listdir(cat_dir)[:max_images_per_class]:


        filepath = os.path.join(cat_dir, filename)

        image = cv2.imread(filepath)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)

        images.append(image)
        labels.append(0)

    # ------------------------
    # Load dog images (label = 1)
    # ------------------------
    print("Loading dog images...")
    for filename in os.listdir(dog_dir)[:max_images_per_class]:

        filepath = os.path.join(dog_dir, filename)

        image = cv2.imread(filepath)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)

        images.append(image)
        labels.append(1)
    
    print("Creating train/test split...")    

    # Convert to NumPy arrays
    X = np.array(images, dtype=np.float32)
    y = np.array(labels)
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state, shuffle=True
    )
    print("Cats:", len(os.listdir(cat_dir)))
    print("Dogs:", len(os.listdir(dog_dir)))

    return X_train, y_train, X_test, y_test
