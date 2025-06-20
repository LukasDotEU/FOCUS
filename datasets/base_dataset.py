import pandas as pd
from torch.utils.data import Dataset

class BaseEEGDataset(Dataset):
    """
    Abstract base class for EEG (and optional Image) datasets.
    Holds a pandas DataFrame `self.metadata` with columns:
      - 'idx'       (int): index of sample in the dataset array
      - 'subject'   (int): subject identifier
      - 'filepath'  (str): filename of EEG trial,
      - 'class_idx' (int): class idx
      - 'image_idx' (int): image idx

    Subclasses must populate `self.metadata` in __init__ and implement:
      - __len__(): return number of samples
      - __getitem__(idx): return a dict with keys:
           'eeg'      : torch.Tensor [C, T]
           'class_idx': int
           'image_idx': int
           'subject'  : int
           'image'    : torch.Tensor or None (if images not used)
           'cwt'      : torch.Tensor or None (if cwt not used)
    """

    def __init__(self, eeg_root: str, images_root: str, use_images: bool, images_file:str, use_cwt: bool):
        super().__init__()
        self.eeg_root = eeg_root
        self.images_root = images_root
        self.use_images = use_images
        self.images_file = images_file
        self.use_cwt = use_cwt

        # Subclasses must create a DataFrame with columns ['idx', 'subject', 'filepath', 'class_idx', 'image_idx']
        self.metadata = pd.DataFrame(columns=['idx', 'subject', 'filepath', 'class_idx', 'image_idx'])

    def __len__(self):
        return len(self.metadata)

    def get_indices_by_subject(self, subject_id):
        """
        Returns a list of dataset indices where subject == subject_id.
        """
        return self.metadata[self.metadata['subject'] == subject_id]['idx'].tolist()

    def get_indices_by_class_idx(self, class_idx):
        """
        Returns a list of dataset indices where class_idx == class_idx.
        """
        return self.metadata[self.metadata['class_idx'] == class_idx]['idx'].tolist()
    
    def get_indices_by_image_idx(self, image_idx):
        """
        Returns a list of dataset indices where image_idx == image_idx.
        """
        return self.metadata[self.metadata['image_idx'] == image_idx]['idx'].tolist()

    def __getitem__(self, idx):
        """
        Must be implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
