def take_from_dataloader():
    dataset = Dataset(root='', data_list_file='data/Datasets/train_new_update.txt', phase='train', input_shape=(1, 128, 128))
    trainloader = data.DataLoader(dataset, batch_size=5)
    for i, (data, label) in enumerate(trainloader):
        id = int(0)
        fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize=(25, 25))
        for plt_ax in ax.flatten():
            new_data = np.transpose(data[id].cpu().detach().numpy(), (1, 2, 0)).squeeze()
            new_data += 127.5
            plt_ax.imshow(new_data, cmap ='gray')
            id += 1
        break
