function struct2txt(folder, res)
%Write struct read from xml to txt
filename = res.annotation.filename;
image_name = filename;
filename = filename(1 : end - 3);
filename = [filename, 'txt'];
filename = fullfile(folder, filename);
file = fopen(filename, 'w');
keys = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', ...
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', ...
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', ...
    'tvmonitor', 'background'};
values = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
class_map = containers.Map(keys, values);
o = res.annotation.object;
for i = 1:1:size(o, 2)
    class_name = o(i).name;
    current_line = image_name;
    current_line = [current_line, ' ', num2str(class_map(class_name))];
    current_line = [current_line, ' ', o(i).bndbox.xmin];
    current_line = [current_line, ' ', o(i).bndbox.ymin];
    current_line = [current_line, ' ', o(i).bndbox.xmax];
    current_line = [current_line, ' ', o(i).bndbox.ymax];
    current_line = [current_line, '\n'];
    fprintf(file, current_line);
end
fclose(file);
