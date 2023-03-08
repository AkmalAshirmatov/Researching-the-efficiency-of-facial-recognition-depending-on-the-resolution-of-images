def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    bbox_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])
            bbox_list.append([int(splits[3]), int(splits[4]), int(splits[5]), int(splits[6])])

        if splits[1] not in data_list:
            data_list.append(splits[1])
            bbox_list.append([int(splits[7]), int(splits[8]), int(splits[9]), int(splits[10])])

    return data_list, bbox_list

def expand2square(pil_img, background_color):
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

def load_image(img_path, bbox):
    image = Image.open(img_path)
    if image is None:
        return None
    image = image.crop(box = bbox)
    image = expand2square(image, (0,0,0))
    image = image.convert('L')
    image = image.resize((128, 128))
    image = T.ToTensor()(image)
    image.float()
    image -= 127.5
    image = image[np.newaxis, :, :]
    return image

def get_featurs(model, test_list, bbox_list, batch_size=10):
    images = None
    features = None
    cnt = 0

    for i, img_path in enumerate(test_list):
        image = load_image(img_path, bbox_list[i])
        if image is None:
            print('read {} error'.format(img_path))
        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)
        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1
            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()
            if features is None:
                features = output
            else:
                features = np.vstack((features, output))
            images = None
    return features, cnt

def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    return fe_dict

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    cnt00 = 0
    cnt11 = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
            cnt00 = np.sum(np.logical_and(y_test == y_true, y_test == 0).astype(int))
            cnt11 = np.sum(np.logical_and(y_test == y_true, y_test == 1).astype(int))

    return (best_acc, best_th, cnt00, cnt11)

def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th, cnt00, cnt11 = cal_accuracy(sims, labels)
    return acc, th, cnt00, cnt11


def lfw_test_verification(model, img_paths, identity_list, bbox_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, bbox_list, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print(f'total verification time is {t}')
    fe_dict = get_feature_dict(identity_list, features)
    acc, th, cnt00, cnt11 = test_performance(fe_dict, compair_list)
    print(f'lfw face verification accuracy: {acc} threshold: {th}
            cnt00 = {cnt00} cnt11 = {cnt11}')
    return acc
