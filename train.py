import os
import argparse
import tensorflow as tf

from models.yolo_v3 import yolo_v3
from models.loss_functions.yolo_v3_loss import yolo_v3_loss
from utils import get_anchors, get_classes, get_training_batch, prepare_data


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-from-checkpoint', type=str,
        help="the path to where a previously trained model's weights are stored")
    parser.add_argument('--class-path', default='utils/coco_classes.txt', type=str,
                        help='the path that points to the class names for the dataset')
    parser.add_argument(
        '--anchors-path', default='utils/anchors.txt', type=str,
        help='the path that points to the anchor values for the models')
    parser.add_argument(
        '--data-path', default='data/image_paths_and_box_info.txt', type=str,
        help='the path that points to the training data text file')
    parser.add_argument(
        '--input-height', default=416, type=int,
        help='the input height of the yolov3 model. The height must be a multiple of 32')
    parser.add_argument(
        '--input-width', default=416, type=int,
        help='The input width of the yolov3 model. The width must be a multiple of 32')
    parser.add_argument(
        '--batch-size', default=32, type=int,
        help='the training batch size')
    parser.add_argument(
        '--max-num-boxes-per-image', default=20, type=int,
        help='the max number of boxes that can be detected within one image')
    parser.add_argument(
        '--num-training-epochs', default=150, type=int,
        help='the number of training epochs')
    parser.add_argument(
        '--learning-rate', default=0.001, type=float,
        help='the learning rate')
    parser.add_argument(
        '--ignore-threshold', default=0.5, type=float,
        help='impacts how the loss is calculated. Must be between zero and one')
    parser.add_argument(
        '--train-val-data-split', default=0.9, type=float,
        help='the split between the data that will be used for training and data that will be used\n\
        for validation')
    parser.add_argument(
        '--train-save-path', default='model-weights/',
        help="the training model's checkpoint save path")
    parser.add_argument(
        '--model-name', default='model.ckpt',
        help='the name that should be given to the checkpoint file')
    parser.add_argument(
        '--tensorboard-save-path', default='tb-logs/tb-train/',
        help='the path where the event files to be used with tensorboard will be saved at')
    parser.add_argument(
        '--test-model-overfit', nargs='?', default=False, type=str2bool, const=True,
        help='this option is useful in testing out if the loss function is working correctly')
    parser.add_argument(
        '--save-iterations', default=100, type=int,
        help="how frequently the model's training weights are saved")
    parser.add_argument(
        '--log-iterations', default=5, type=int,
        help="how frequently the model's loss is logged for it to be inspected in Tensorboard")
    args = parser.parse_args()

    # args info
    print('[i] checkpoint: ', args.train_from_checkpoint)
    print('[i] path of class file: ', args.class_path)
    print('[i] path of anchors file: ', args.anchors_path)

    # read inputs
    h = args.input_height
    w = args.input_width
    ignore_thresh = args.ignore_threshold
    max_num_boxes_per_image = args.max_num_boxes_per_image
    anchors = get_anchors(args.anchors_path)

    lr = args.learning_rate
    num_anchors_per_detector = len(anchors) // 3
    num_detectors_per_image = num_anchors_per_detector * (
            ((h / 32) * (w / 32)) + ((h / 16) * (w / 16)) + ((h / 8) * (w / 8)))

    class_names = get_classes(args.class_path)
    num_classes = len(class_names)

    # tensorboard path
    tb_train_path = args.tensorboard_save_path + 'train/'
    tb_val_path = args.tensorboard_save_path + 'val/'
    training_data, validation_data, batch_size = prepare_data(
        args.data_path,
        args.train_val_data_split,
        args.batch_size,
        args.test_model_overfit)

    tf.reset_default_graph()

    # build graph
    with tf.variable_scope('y_true'):
        y_true_data = tf.placeholder(dtype=tf.float32, shape=[None, num_detectors_per_image, num_classes + 5])
    with tf.variable_scope('y_true_boxes'):
        y_true_box_data = tf.placeholder(dtype=tf.float32,
                                         shape=[None, max_num_boxes_per_image * num_anchors_per_detector, 4])
    with tf.variable_scope('x_input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, h, w, 3])

    yolo_outputs = yolo_v3(inputs=X, num_classes=len(class_names), anchors=anchors, h=h, w=w, training=True)  # output
    loss = yolo_v3_loss(yolo_outputs, y_true_data, y_true_box_data, ignore_threshold=ignore_thresh, anchors=anchors,
                        num_classes=num_classes, h=h, w=w, batch_size=batch_size)

    tf.summary.scalar('loss', loss)
    global_step = tf.get_variable(name='global_step', trainable=False, initializer=0, dtype=tf.int32)

    # returns a varlist containing only the vars of the conv layers right before the yolo layers
    trainable_var_list = tf.trainable_variables()
    last_layer_var_list = [i for i in trainable_var_list if i.shape[-1] == (5 + num_classes) * num_anchors_per_detector]
    train_op_with_frozen_variables = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step,
                                                                                       var_list=last_layer_var_list)
    train_op_with_all_variables = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step,
                                                                                    var_list=trainable_var_list)
    summ = tf.summary.merge_all()

    # info
    print('-------info-------')
    print('[i] model weights will be saved with filename: ', args.model_name)
    print('[i] tensorboard event files located at path: ', args.tensorboard_save_path)
    # build training loop
    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(tb_train_path, sess.graph)
        val_writer = tf.summary.FileWriter(tb_val_path)

        # initialize model weights either randomly or from a saved checkpoint
        saver = tf.train.Saver()
        if args['train_from_checkpoint'] is None:
            print('[i] initializing variables...')
            sess.run(tf.global_variables_initializer())
        else:
            print('[i] restoring weights from checkpoint: ', args.train_from_checkpoint)
            saver.restore(sess, args.train_from_checkpoint)

        num_iterations = args.num_epochs * len(training_data)

        print('[i] beginning to train the model...')
        for i in range(num_iterations):

            input_images, y_true, y_true_boxes = get_training_batch(training_data, anchors, num_classes,
                                                                    batch_size=batch_size, h=h, w=w,
                                                                    random=not args.test_model_overfit)

            # For the first epochs, train with the frozen layers. Then, unfreeze the entire graph.
            if i < num_iterations // 3:
                sess.run(train_op_with_frozen_variables,
                         feed_dict={X: input_images, y_true_data: y_true, y_true_box_data: y_true_boxes})
            else:
                sess.run(train_op_with_all_variables,
                         feed_dict={X: input_images, y_true_data: y_true, y_true_box_data: y_true_boxes})

            if i % args.log_iterations == 0:
                # write the training loss to tensorboard
                lt, st = sess.run([loss, summ],
                                  feed_dict={X: input_images, y_true_data: y_true, y_true_box_data: y_true_boxes})
                train_writer.add_summary(st, i)

                # write the validation loss to tensorboard if we are not in overfit mode
                if not args.test_model_overfit:
                    input_images, y_true, y_true_boxes = get_training_batch(validation_data, anchors, num_classes,
                                                                            batch_size=batch_size, h=h, w=w,
                                                                            random=not args.test_model_overfit)
                    lv, sv = sess.run([loss, summ],
                                      feed_dict={X: input_images, y_true_data: y_true, y_true_box_data: y_true_boxes})
                    val_writer.add_summary(sv, i)
                    print(
                        "| iteration: " + str(i) + ", training loss: " + str(
                            round(lt, 2)) + ", validation loss: " + str(
                            round(lv, 2)))
                else:
                    print("| iteration: " + str(i) + ", training loss: " + str(round(lt, 2)))

            if i % args.save_iterations == 0:
                print('[i] saving model weights at path: ', args.train_save_path)
                saver.save(sess, os.path.join(args.train_save_path, args.model_name), global_step)

            print('################################################################################')

    train_writer.close()
    val_writer.close()


if __name__ == '__main__':
    _main()
