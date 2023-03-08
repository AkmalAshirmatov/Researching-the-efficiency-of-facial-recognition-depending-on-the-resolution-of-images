if __name__ == '__main__':

    ROOT_FOLDER = "/Users/akmal/Downloads/lfw-deepfunneled/"

    last_person = ""
    timer = int(0)
    all_names = []

    train = open("/Users/akmal/Desktop/python_data/train.txt", "w")
    test = open("/Users/akmal/Desktop/python_data/test.txt", "w")
    test_verificaton = open("/Users/akmal/Desktop/python_data/test_verification.txt", "w")

    cnt_images_train = int(0)
    cnt_images_test = int(0)

    arrays_of_images = []
    for path in glob.iglob(os.path.join(ROOT_FOLDER, "**", "*.jpg")):

        person = path.split("/")[-2]

        if person != last_person:
            if last_person != "" and len(all_names) > 2:
                sz = len(all_names)
                all_names = np.random.permutation(all_names)
                array_person = []
                for i in range(sz):
                    if (sz > 3 and i * 100 // sz < 75) or (sz == 3 and i * 100 // sz < 66):
                        train.write(all_names[i] + " " + str(timer) + "\n")
                        cnt_images_train += 1
                    else:
                        test.write(all_names[i] + " " + str(timer) + "\n")
                        cnt_images_test += 1
                        array_person.append(all_names[i])
                arrays_of_images.append(array_person)
                timer += 1
            all_names = []
            last_person = person

        mypath = path.split("/")[3:]
        split_path = path.split("/")
        s = "/content/gdrive/MyDrive/arcface-pytorch/data/Datasets/
            lfw-deepfunneled/" + split_path[-2] + "/" + split_path[-1]
        all_names.append(s)

    print("cnt_images_train =", cnt_images_train)
    print("cnt_images_test =", cnt_images_test)
    print("cnt_people =", timer)

    for i in range(750):
        id = np.random.randint(0, timer)
        while len(arrays_of_images[id]) == 1:
            id = np.random.randint(0, timer)
        sz = len(arrays_of_images[id])
        id_inside1 = np.random.randint(0, sz)
        id_inside2 = np.random.randint(0, sz)
        if id_inside1 == id_inside2:
            id_inside2 = (id_inside2 + 1) % sz
        if id_inside1 == id_inside2:
            print("bad")
        test_verificaton.write(arrays_of_images[id][id_inside1] + " " + arrays_of_images[id][id_inside2] + " 1\n")

    for i in range(750):
        id1 = np.random.randint(0, timer)
        id2 = np.random.randint(0, timer)
        if id1 == id2:
            id2 = (id2 + 1) % timer
        if id1 == id2:
            print("bad")
        id_inside1 = np.random.randint(0, len(arrays_of_images[id1]))
        id_inside2 = np.random.randint(0, len(arrays_of_images[id2]))
        test_verificaton.write(arrays_of_images[id1][id_inside1] + " " + arrays_of_images[id2][id_inside2] + " 0\n")
