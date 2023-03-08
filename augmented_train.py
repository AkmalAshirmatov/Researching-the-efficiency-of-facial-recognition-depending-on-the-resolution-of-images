def form_string_from_bbox(bbox):
    return str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3])

def add_bbox_and_angle_train(model):
    list_of_strings = open("data/Datasets/train.txt", "r")
    list_of_strings = list_of_strings.read()
    list_of_strings = list_of_strings.split("\n")
    list_of_strings.pop(-1)
    train = open("data/Datasets/train_new_update.txt", "w")
    degrees = [-30, 30]
    for iter in range(0, 5):
        for id, path_id in enumerate(list_of_strings):
            while 1:
                angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
                if iter == 4:
                    angle = 0
                path = path_id.split(" ")[-2]
                pil_image = Image.open(path)
                pil_image = TF.rotate(pil_image, angle)
                image = np.array(pil_image).copy()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                annotation = model.predict_jsons(image)
                bbox = annotation[0]['bbox']
                if len(bbox) != 4:
                    continue
                train.write(path_id + " " + form_string_from_bbox(bbox) + " " + str(angle) + "\n")
                break
            if id % 100 == 0:
                print(id)
    train.close()

if __name__ == '__main__':

    model = get_model("resnet50_2020-07-20", max_size=250)
    model.eval()

    add_bbox_and_angle_train(model)
