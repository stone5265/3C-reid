from torch.utils.data import Dataset
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, transform=None, load_img=True):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.load_img = load_img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        img_path, pid, cam = self.dataset[indices]
        idx = None
        if isinstance(pid, list):
            pid, idx = pid

        if self.load_img:
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            if idx is not None:
                return img, [pid, idx], cam
            else:
                return img, pid, cam

        return pid, cam
