import os, pprint
from black import ipynb_diff
from natsort import natsorted
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

import cv2, copy
import numpy as np
cv2.namedWindow('img', cv2.WINDOW_NORMAL)



class ImageMeanANDStd:
    def __init__(self):
        self.step    = 25
        self.width   = 64
        self.height  = 64
        self.channle = 3


    def load_dataset(self, repository):
        self.save_dir = repository.dataset_save_dir
        db_files      = os.listdir(repository.dataset_save_dir)
        db_files      = [db for db in db_files if '.db' in db]
        db_files      = natsorted(db_files)
        num_files     = len(db_files)
        images        = np.random.randn(num_files, self.step, self.width, self.height, self.channle)
        pprint.pprint(db_files)
        for index, db in enumerate(db_files):
            db_name, suffix = db.split(".")
            repository.open(filename=db_name)
            img_can = repository.repository["image"]["canonical"]
            img_ran = repository.repository["image"]["random_nonfix"]
            step, width, height, channel = img_can.shape
            for t in range(step):
                print("({}/{}) [{}] step: {}".format(index+1, num_files, db, t))

                # cv2.imshow('img', np.concatenate((img_ran[t], img_can[t]), axis=1))
                # cv2.waitKey(10)
                images[index, t] = copy.deepcopy(img_can[t])
            repository.close()
        return images


    def save_dist(self, images, title: str, filename: str):
        fig, ax = plt.subplots()
        ax.hist(images.ravel(), bins=50, density=True)
        ax.set_xlabel("pixel values")
        ax.set_ylabel("relative frequency")
        ax.set_title(title)
        fig.savefig(repository.dataset_save_dir + "/" + filename)


if __name__ == '__main__':
    from scipy.special import boxcox1p
    from scipy.stats import boxcox, yeojohnson
    # from spicy.stats import boxcox_normmax

    repository = Repository(
        dataset_dir  = "./dataset",
        dataset_name = "dataset_202210221514_valve2000_train",
    )

    imgmeanstd = ImageMeanANDStd()
    img_origin = imgmeanstd.load_dataset(repository)
    print("img_origin.shape = ", img_origin.shape)

    img_origin_01 = img_origin / 255.
    print("img_origin_01 minmax = [{}, {}]", img_origin_01.min(), img_origin_01.max())

    img_origin_01_0 = img_origin_01[:, :, :, :, 0]
    img_origin_01_1 = img_origin_01[:, :, :, :, 1]
    img_origin_01_2 = img_origin_01[:, :, :, :, 2]

    print("-----------------------------------------")
    print("mean: [r, g, b] = [{}, {}, {}]".format(img_origin_01_0.mean(), img_origin_01_1.mean(), img_origin_01_2.mean()))
    print(" std: [r, g, b] = [{}, {}, {}]".format(img_origin_01_0.std(),  img_origin_01_1.std(),  img_origin_01_2.std()))
    print("-----------------------------------------")

    img_converted_r = (img_origin_01_0 - img_origin_01_0.mean()) / img_origin_01_0.std()
    img_converted_g = (img_origin_01_1 - img_origin_01_1.mean()) / img_origin_01_1.std()
    img_converted_b = (img_origin_01_2 - img_origin_01_2.mean()) / img_origin_01_2.std()

    # import ipdb; ipdb.set_trace()
    imgmeanstd.save_dist(img_origin[:, :, :, :, 0], title="original dataset distribution\n({})".format(repository.dataset_name), filename="pixel_dist_{}_r".format(repository.dataset_name))
    imgmeanstd.save_dist(img_origin[:, :, :, :, 1], title="original dataset distribution\n({})".format(repository.dataset_name), filename="pixel_dist_{}_g".format(repository.dataset_name))
    imgmeanstd.save_dist(img_origin[:, :, :, :, 2], title="original dataset distribution\n({})".format(repository.dataset_name), filename="pixel_dist_{}_b".format(repository.dataset_name))

    imgmeanstd.save_dist(img_converted_r, title="preprocessed dataset distribution\n({})".format(repository.dataset_name), filename="pixel_dist_{}_preprocess_r".format(repository.dataset_name))
    imgmeanstd.save_dist(img_converted_g, title="preprocessed dataset distribution\n({})".format(repository.dataset_name), filename="pixel_dist_{}_preprocess_g".format(repository.dataset_name))
    imgmeanstd.save_dist(img_converted_b, title="preprocessed dataset distribution\n({})".format(repository.dataset_name), filename="pixel_dist_{}_preprocess_b".format(repository.dataset_name))
