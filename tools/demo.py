import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network


CLASSES = ('__background__', # always index 0
                         'apple', 'ball', 'banana', 'bell_pepper',
                         'binder', 'bowl', 'calculator', 'camera', 'cap',
                         'cell_phone', 'cereal_box', 'coffe_mug', 'comb', 'dry_battery',
                         'flashlight', 'food_bag', 'food_box', 'food_can',
                         'food_cup', 'food_jar', 'garlic',
                         'glue_stick', 'greens', 'hand_towel', 'instant_noddles',
                         'keyboard', 'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 
                         'notebook', 'onion', 'orange' 'peach', 'pear', 'pitcher',
                         'plate', 'pliers', 'potato', 'rubber_eraser', 'scissors', 'shampoo',
                         'soda_can', 'sponge', 'stapler', 'tomato', 'toothbrush', 'toothpaste', 'water_bottle')


#CLASSES = ('__background__','person','bike','motorbike','car','bus')
def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im.copy()

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

    #    ax.add_patch(
    #        plt.Rectangle((bbox[0], bbox[1]),
    #                      bbox[2] - bbox[0],
    #                      bbox[3] - bbox[1], fill=False,
    #                      edgecolor='red', linewidth=3.5)
    #        )
    #    ax.text(bbox[0], bbox[1] - 2,
    #            '{:s} {:.3f}'.format(class_name, score),
    #            bbox=dict(facecolor='blue', alpha=0.5),
    #            fontsize=14, color='white')

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
        m=im.copy()
        cv2.rectangle(m,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0))
        im=m
        print('detected shit')
    #plt.axis('off')
    #plt.tight_layout()
    #plt.savefig('out%d.png'%THECOUNT)
    return im.copy()

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file =  image_name
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.03
    NMS_THRESH = 0.2
    for cls_ind, cls in enumerate(CLASSES[1:]):
        print(cls_ind)
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        im = vis_detections(im.copy(), cls, dets, cls_ind, thresh=CONF_THRESH)

    cv2.imwrite('out'+image_name[10:20]+'.jpg',im)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    THECOUNT=0

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))
        
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    im_names = ['../14881375_10205614413678464_443157071_o.jpg','../14894523_10205614414158476_1426895018_o.jpg','../14922941_10205614414238478_911648154_o.jpg','../14923047_10205614413958471_1613650859_o.jpg',
                '../14923932_10205614414078474_1863774729_o.jpg','../14954399_10205614414198477_1616147118_o.jpg','../250px-Batteries.jpg']


    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        print (im_name)
        demo(sess, net, im_name)

    plt.show()

