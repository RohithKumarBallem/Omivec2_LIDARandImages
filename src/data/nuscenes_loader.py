import os
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

class NuScenesMiniSample:
    def __init__(self, dataroot, version, cams, lidar):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.cams = list(cams)
        self.lidar = str(lidar)

    def get_first_sample(self):
        scene = self.nusc.scene[0]
        sample_token = scene["first_sample_token"]
        sample = self.nusc.get("sample", sample_token)

        imgs = {}
        for cam in self.cams:
            sd = self.nusc.get("sample_data", sample["data"][cam])
            img_path = os.path.join(self.nusc.dataroot, sd["filename"])
            img = cv2.imread(img_path)
            imgs[cam] = img

        sd_lidar = self.nusc.get("sample_data", sample["data"][self.lidar])
        lidar_path = os.path.join(self.nusc.dataroot, sd_lidar["filename"])
        pc = LidarPointCloud.from_file(lidar_path)
        points = pc.points.T  # (N,4)
        return sample_token, imgs, points
