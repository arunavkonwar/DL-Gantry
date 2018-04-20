import glob
import skimage.io
import skimage.exposure
import numpy as np
import matplotlib.pyplot as plt
import os


root_dir = "/home/arunav/code-new/binary/generated_images_8k"
names = np.array(glob.glob(os.path.join(root_dir, "*.jpg")))
indexes = np.array([int(os.path.splitext(os.path.basename(name))[0]) for name in names])
argsort_indexes = np.argsort(indexes)
names = names[argsort_indexes]
names = names[:10]
print(names)
images = [skimage.io.imread(name) for name in names]
images = [skimage.exposure.rescale_intensity(image * 1.0, out_range=np.float32) for image in images]
#plt.imshow(images[0], cmap="gray")
#plt.show()
#print(images[0].shape, images[0])

