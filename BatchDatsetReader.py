"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
from tqdm import tqdm


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0
    name = ""
    def __init__(self, name, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        self.name = name
        print("Initializing Batch Dataset Reader, It may take minutes...")
        print(image_options)
        self.files = records_list
        # print("files:", self.files)
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        # 1.
        self.__channels = True
        # self.images = np.array([self._transform(filename['image']) for filename in self.files])
        # to display the progress info to users
        # 读取训练集图像         
        # # 扫描files字典中所有image 图片全路径        
        # # 根据文件全路径读取图像，并将其扩充为RGB格式         
        # # 遍历self.files中每一个元素赋值给filename，再经过self._transform(filename['image'])获得一个新的值，以列表形式返回         
        # # 遍历self.files中每一个元素赋值给filename，再将filename中'image'对应的值带入self._transform，返回值给np.array  
        self.images = np.array([self._transform(filename['image']) for filename in tqdm(self.files)])
        self.__channels = False
        # 读取label的图像，由于label图像是二维的，这里需要扩展为三维
        self.annotations = np.array([np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in tqdm(self.files)])
        print("image.shape:", self.images.shape)
        print("annotations.shape:", self.annotations.shape)

    """
        resize images to fixed resolution for the DNN
    """
    # 把图像转为 numpy数组     # 输入的是文件路径
    def _transform(self, filename):
        # 1. read image
        image = misc.imread(filename)
        # 2. make sure it is RGB image          # 应该是怕有一维的灰度图像，这样重复3次让它强行变成3维数组，             # 将图片三个通道设置为一样的图片
        if self.__channels and len(
                image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])
        # 3. resize it
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(
                image, [resize_size, resize_size], interp='nearest')            # 使用最近邻插值法resize图片
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):           # self.batch_offset 循环累加，实现了每次只取batch_size张图，全部取完后，打乱顺序重新开始
        start = self.batch_offset               # 读取下一个batch  所有offset偏移量+batch_size

        self.batch_offset += batch_size         # iamges存储所有图片信息 images.shape(len, h, w)
        if self.name == "logs":
            self.batch_offset = 4
        if self.batch_offset > self.images.shape[0]:        # 如果下一个batch的偏移量超过了图片总数 说明完成了一个epoch
            # Finished epoch
            self.epochs_completed += 1
            if self.name == "train":
                print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])          # arange生成数组(0 - len-1) 获取图片索引        即所有训练图片的数量
            np.random.shuffle(perm)                         #对perm洗牌，打乱顺序
            self.images = self.images[perm]                 # 洗牌之后的图片顺序
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]              # 取出当前batch

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]

    def get_num_of_records(self):
        return self.images.shape[0]
