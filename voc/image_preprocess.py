from PIL import Image
import os

class ImgTool:
    def __init__(self, path):
        self.load_path = path

    def get_image(self, anns, show = False, crop = True):
        if len(anns > 100):
            print("This function is just for show sample image.")
            return
        data = list()
        path = self.load_path
        for ann in anns:
            image = Image.open('{}{}'.format(path, ann[-1]))
            if(crop):
                image = image.crop(ann[1])
            if(show):
                image.show()
            data.append(image)
        return data


    def save_image(self, anns, path):
        if not os.path.exists(path):
            os.mkdir(path)
            print("Path is created!")
        length = len(anns)
        load_path = self.load_path
        for i in range(length):
            image = Image.open('{}{}'.format(load_path, anns[i][-1]))
            filename = '{}{}{}'.format(path, i, '.jpg')
            image = image.crop(anns[i][1])
            image.save(filename, 'jpeg')
            if(i % 100 == 0):
                print(i / length)
