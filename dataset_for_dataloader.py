class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(1, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape
        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        self.imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

    def __getitem__(self, index):
        transform = T.Compose([
            T.ToTensor(),
        ])
        splits = self.imgs[index].split()
        image = self.open_image(index)
        image = self.squeeze(image, 1)
        image = self.augmentation(image)
        image = image.resize(self.input_shape[1:])
        image = transform(image)
        image -= 127.5
        label = np.int32(splits[1])
        return image.float(), label

    def __len__(self):
        return len(self.imgs)

    def squeeze(self, image, coef):
        width, height = image.size
        image = image.resize((width // coef, height // coef))
        return image

    def augmentation(self, image):
        image = T.RandomHorizontalFlip()(image)
        return image

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def get_bbox(self, index):
        list_of_strings = self.imgs[index].split(' ')
        bbox = [int(list_of_strings[2]), int(list_of_strings[3]), int(list_of_strings[4]), int(list_of_strings[5])]
        return bbox

    def open_image(self, index):
        bbox = self.get_bbox(index)
        image = Image.open(self.imgs[index].split(' ')[0])
        angle = float(self.imgs[index].split(' ')[6])
        image = TF.rotate(image, angle)
        image = image.crop(box = bbox)
        image = self.expand2square(image, (0,0,0))
        image = image.convert('L')
        return image
