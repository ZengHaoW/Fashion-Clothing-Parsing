from __future__ import print_function

import function_definitions as fd
import BatchDatsetReader as DataSetReader
import read_10k_data as fashion_parsing
import read_CFPD_data as ClothingParsing
import read_LIP_data as HumanParsing
import TensorflowUtils as Utils

import numpy as np
import tensorflow as tf

# Hide the warning messages about CPU/GPU
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


DATA_SET = "CFPD"
# DATA_SET = "LIP"

FLAGS = tf.flags.FLAGS

if DATA_SET == "CFPD":
    tf.flags.DEFINE_integer("batch_size", "31", "batch size for training")
    tf.flags.DEFINE_integer(
        "training_epochs",
        "25",
        "number of epochs for training")
    tf.flags.DEFINE_string("logs_dir", "logs/FCN_CFPD/",
                           "path to logs directory")
    tf.flags.DEFINE_string(
        "data_dir", "C:/CFPD/", "path to dataset")

if DATA_SET == "LIP":
    tf.flags.DEFINE_integer("batch_size", "16", "batch size for training")
    tf.flags.DEFINE_integer(
        "training_epochs",
        "50",
        "number of epochs for training")
    tf.flags.DEFINE_string("logs_dir", "logs/FCN_LIP/",
                           "path to logs directory")
    tf.flags.DEFINE_string(
        "data_dir", "C:/Users/zx08x/Desktop/new/LIP/", "path to dataset")

tf.flags.DEFINE_float(
    "learning_rate",
    "1e-5",
    "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
# tf.flags.DEFINE_string('mode', "test", "Mode train/ test/ visualize")
# tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1001)

NUM_OF_CLASSES = 18  # Upper-lower cloth parsing # Dressup 10k
if DATA_SET == "CFPD":
    NUM_OF_CLASSES = 23  # Fashion parsing 23 # CFPD
if DATA_SET == "LIP":
    NUM_OF_CLASSES = 20  # human parsing # LIP

IMAGE_SIZE = 224
DISPLAY_STEP = 300
TEST_DIR = FLAGS.logs_dir + "TestImage/"
VIS_DIR = FLAGS.logs_dir + "VisImage/"

"""
   Train, Test
"""


# 预测流程，image是输入图像，keep_prob dropout比例
def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    # 1. donwload VGG pretrained model from network if not did before
    #    model_data is dictionary for variables from matlab mat file
    print("setting up vgg initialized conv layers ...")
    model_data = Utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]                         #通过字典获取mean值,vgg模型参数里有normaliza这个字典，三个0用来去虚维找到mean 
    mean_pixel = np.mean(mean, axis=(0, 1))             

    weights = np.squeeze(model_data['layers'])                          # 压缩VGG网络中参数，把维度是1的维度去掉 剩下的就是权重

    processed_image = Utils.process_image(image, mean_pixel)            #预处理函数 return image - mean_pixel # 图像减平均值实现标准化

    # 2. construct model graph
    with tf.variable_scope("inference"):            # tf.variable_scope() 主要结合 tf.get_variable() 来使用，实现变量共享。inference相当于变量名前缀 
        # 2.1 VGG
        image_net = fd.vgg_net(weights, processed_image)        # 传入权重参数和预测图像，获得所有层输出结果
        conv_final_layer = image_net["conv5_3"]                 # 获得输出结果
        
        pool5 = Utils.max_pool_2x2(conv_final_layer)

        W6 = Utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = Utils.bias_variable([4096], name="b6")
        conv6 = Utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            Utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)           #防止或减轻过拟合而使用的函数，它一般用在全连接层。keep_prob: 设置神经元被选中的概率

        W7 = Utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = Utils.bias_variable([4096], name="b7")
        conv7 = Utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            Utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = Utils.weight_variable([1, 1, 4096, NUM_OF_CLASSES], name="W8")
        b8 = Utils.bias_variable([NUM_OF_CLASSES], name="b8")
        conv8 = Utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = Utils.weight_variable(
            [4, 4, deconv_shape1[3].value, NUM_OF_CLASSES], name="W_t1")
        b_t1 = Utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = Utils.conv2d_transpose_strided(
            conv8, W_t1, b_t1, output_shape=tf.shape(
                image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = Utils.weight_variable(
            [4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = Utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = Utils.conv2d_transpose_strided(
            fuse_1, W_t2, b_t2, output_shape=tf.shape(
                image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack(
            [shape[0], shape[1], shape[2], NUM_OF_CLASSES])
        W_t3 = Utils.weight_variable(
            [16, 16, NUM_OF_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = Utils.bias_variable([NUM_OF_CLASSES], name="b_t3")
        conv_t3 = Utils.conv2d_transpose_strided(
            fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
                   # 目前conv_t3的形式为size为和原始图像相同的size，通道数与分类数相同         
                   # # 这句我的理解是对于每个像素位置，根据第3维度（通道数）通过argmax能计算出这个像素点属于哪个分类         
                   # # 也就是对于每个像素而言，NUM_OF_CLASSESS个通道中哪个数值最大，这个像素就属于哪个分类         
                   # # 每个像素点有21个值，哪个值最大就属于那一类        
                   # # 返回一张图，每一个点对于其来别信息shape=[b,h,w]
        # prob = tf.nn.softmax(conv_t3, axis =3)
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3, image_net


"""inference
  optimize with trainable paramters (Check which ones)
  loss_val : loss operator (mean(
"""


def train(loss_val, var_list, global_step):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            Utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads, global_step=global_step)


def main(argv=None):
    # 1. input placeholders
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    image = tf.placeholder(tf.float32, shape = (None, IMAGE_SIZE, IMAGE_SIZE, 3), name = "input_image")
    annotation = tf.placeholder(tf.int32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1), name="annotation")
    # global_step = tf.Variable(0, trainable=False, name='global_step')

    # 2. construct inference network
     # 预测一个batch图像  获得预测图[b,h,w,c=1]  结果特征图[b,h,w,c=151]
    pred_annotation, logits, net = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=3)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=3)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=3)

    # 3. loss measure       # 空间交叉熵损失函数[b,h,w,c=151]  和labels[b,h,w]    每一张图分别对比
    loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.squeeze(
                annotation,
                squeeze_dims=[3]),
            name="entropy")))
    tf.summary.scalar("entropy", loss)

    # 4. optimizing
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            Utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var, net['global_step'])

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader from ", FLAGS.data_dir, "...")
    print("data dir:", FLAGS.data_dir)

    #train_records, valid_records = fashion_parsing.read_dataset(FLAGS.data_dir)
    #test_records = None
    if DATA_SET == "CFPD":
        train_records, valid_records, test_records, logs_records = ClothingParsing.read_dataset(
            FLAGS.data_dir)
    if DATA_SET == "LIP":
        train_records, valid_records = HumanParsing.read_dataset(
            FLAGS.data_dir)
    print("test_records length :", len(test_records))
    print("train_records length :", len(train_records))
    print("valid_records length :", len(valid_records))
    print("logs_records length :", len(logs_records))
    print("Setting up dataset reader")
    train_dataset_reader = None
    validation_dataset_reader = None
    test_dataset_reader = None
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}

    if FLAGS.mode == 'train':
        train_dataset_reader = DataSetReader.BatchDatset("train", train_records, image_options)
        validation_dataset_reader = DataSetReader.BatchDatset("val", valid_records, image_options)
        logs_dataset_reader = DataSetReader.BatchDatset("logs", logs_records, image_options)

    if FLAGS.mode == 'visualize':
        validation_dataset_reader = DataSetReader.BatchDatset(
            valid_records, image_options)
    if FLAGS.mode == 'test' or FLAGS.mode == 'crftest' or FLAGS.mode == 'predonly' or FLAGS.mode == "fulltest":
        if DATA_SET == "CFPD":
            test_dataset_reader = DataSetReader.BatchDatset("test",
                test_records, image_options)
        else:
            test_dataset_reader = DataSetReader.BatchDatset(
                valid_records, image_options)
            test_records = valid_records

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=4)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    # 5. parameter setup

    # 5.1 init params
    sess.run(tf.global_variables_initializer())

    # 5.2 restore params if possible
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # 6. train-mode
    if FLAGS.mode == "train":

        fd.mode_train(sess, FLAGS, net, train_dataset_reader, validation_dataset_reader, logs_dataset_reader, train_records, logs_records,pred_annotation,
                      image, annotation, keep_probability, logits, train_op, loss, summary_op, summary_writer, saver,
                      DISPLAY_STEP)

    elif FLAGS.mode == "test":  # heejune added

        fd.mode_new_test(sess, FLAGS, TEST_DIR, test_dataset_reader, test_records,
                         pred_annotation, image, annotation, keep_probability, logits, NUM_OF_CLASSES)

        # fd.mode_test(sess, FLAGS, TEST_DIR, test_dataset_reader, test_records,
        # pred_annotation, image, annotation, keep_probability, logits, NUM_OF_CLASSES)

    elif FLAGS.mode == "view":
        train_dataset_reader = DataSetReader.BatchDatset("train", train_records, image_options)
        logs_dataset_reader = DataSetReader.BatchDatset("logs", logs_records, image_options)
        fd.mode_view(sess, FLAGS, "./VisImage/", train_dataset_reader, logs_dataset_reader,
                      logs_records,
                      pred_annotation, image, annotation, keep_probability, logits, 23)
    '''
    elif FLAGS.mode == "crftest":

        fd.mode_predonly(sess, FLAGS, TEST_DIR, test_dataset_reader, test_records,
                         pred_annotation, image, annotation, keep_probability, logits, NUM_OF_CLASSES)

    elif FLAGS.mode == "predonly":

        fd.mode_predonly(sess, FLAGS, TEST_DIR, test_dataset_reader, test_records,
                         pred_annotation, image, annotation, keep_probability, logits, NUM_OF_CLASSES)

    elif FLAGS.mode == "fulltest":

        fd.mode_full_test(sess, FLAGS, TEST_DIR, test_dataset_reader, test_records,
                          pred_annotation, image, annotation, keep_probability, logits, NUM_OF_CLASSES)
    '''
    sess.close()


if __name__ == "__main__":
    tf.app.run()
