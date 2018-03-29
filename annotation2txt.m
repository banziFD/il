% Build metadata before train

clear; clc;
% set location

path_dataset = '/home/spyisflying/dataset/voc/VOC2012';
addpath('./VOCTool/VOCcode');

path_annotation = fullfile(path_dataset, 'Annotations');
path_image = fullfile(path_dataset, 'JPEGImages');

ann_files = dir(path_annotation);
ann_files = ann_files(3:end);
img_files = dir(path_image);
img_files = img_files(3:end);

for i = 1:1:size(ann_files, 1)
    current_file = fullfile(path_annotation, ann_files(i).name);
    res = VOCreadxml(current_file);
    struct2txt('./annotations', res);
end
