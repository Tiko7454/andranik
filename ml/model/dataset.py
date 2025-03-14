import os
from torch.utils.data import Dataset
import cv2


class ChessPiecesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in subdirectories,
                                each corresponding to a class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # List all class directories in the root directory
        self.class_names = os.listdir(root_dir)
        self.class_names.sort()  # Sort to maintain consistent class indexing

        # Create a mapping from class name to class index
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.ind_to_class = {idx: class_name for idx, class_name in enumerate(self.class_names)}

        # List all image paths along with their corresponding class labels
        self.img_paths = []
        self.labels = []

        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.endswith(('jpg', 'jpeg', 'png')):  # Filter for image files
                        self.img_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # Open the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        return img, label
