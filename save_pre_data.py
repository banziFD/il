from ann_preprocess import AnnTool
from image_preprocess import ImgTool

# get annotations
path = "./annotations/"

annotations = AnnTool(path).get_annotations()

# keys = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#     'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
#     'tvmonitor', 'background']
# values = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# class_chart = dict([(keys[i], values[i]) for i in range(len(keys))])
# annotations = AnnTool(path).get_annotations(class_chart['aeroplane'])
AnnTool.save(annotations, 'anns')
# save images
img_tool = ImgTool('/home/spyisflying/dataset/voc/VOC2012/JPEGImages/')
img_tool.save_image(annotations, './images/')