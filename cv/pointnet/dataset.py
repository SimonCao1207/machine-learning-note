import transform as tf
from torchvision import transforms
from pathlib import Path
from torch.utils.data import Dataset
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELNET10_PATH = Path(os.path.join(FILE_PATH, "data/ModelNet10"))
if not os.path.exists(MODELNET10_PATH):
    assert("Cannot find path to data")

class ModelNetDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        folders = [dir for dir in sorted(os.listdir(root)) if os.path.isdir(root /dir)]
        self.classes = {f:i for i,f in enumerate(folders)}
        self.files= []
        default_transform = transforms.Compose([
            tf.PointSampler(1024),
            tf.Normalize(),
            tf.ToTensor()
        ])
        self.transforms  = transform if transform != None else default_transform
        for cls, idx  in self.classes.items():
            data_dir = root / Path(cls) / split
            for f in os.listdir(data_dir):
                if f.endswith(".off"):
                    sample = {
                        "path" : data_dir / f,
                        "category" : cls
                    }
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def _read_data(self, f):
        if 'OFF' != f.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, __ = tuple([int(s) for s in f.readline().strip().split(' ')])
        verts = [[float(s) for s in f.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in f.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        return verts, faces

    def __getitem__(self, idx):
        path, category = self.files[idx]["path"], self.files[idx]["category"]
        with open(path, 'r') as f:
            verts, faces = self._read_data(f)
            pointcloud = self.transforms((verts, faces))
        return {
            "pointcloud" : pointcloud,
            "category" : category
        }


if __name__ == "__main__":
    d = ModelNetDataset(MODELNET10_PATH)
    pc = d[0]["pointcloud"]
    print(pc.shape)

