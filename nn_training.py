def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    print("saved", save_name)
    return save_name

if __name__ == '__main__':
    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")
    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  num_workers=opt.num_workers, shuffle = True)
    print(f'{len(trainloader)} train iters per epoch:')
    criterion = FocalLoss(gamma=2)
    model = resnet_face18(use_se=opt.use_se)
    metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)
    start = time.time()
    identity_list, bbox_list = get_lfw_list(opt.lfw_test_list)
    img_paths = identity_list
    lfw_test_acc = []
    loss_mean = []
    acc_mean = []
    for i in range(opt.max_epoch):
        if i > 0:
          scheduler.step()

        model.train()
        print("epoch = ", i)
        loss_epoch = []
        acc_epoch = []
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            loss_epoch.append(loss.item())
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc_epoch.append(np.mean((output == label).astype(float)))

            if iters % opt.print_freq == 0:
                acc = np.mean((output == label).astype(float))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print(f'{time_str} epoch {i} iter {ii} {speed} iters/s
                        loss {loss.item()}, acc {acc}')
                start = time.time()

        model.eval()

        my_acc = lfw_test_verification(model=model, img_paths=img_paths, identity_list=identity_list, bbox_list=bbox_list, compair_list=opt.lfw_test_list, batch_size=opt.test_batch_size)
        lfw_test_acc.append(my_acc)
        loss_mean.append(np.mean(loss_epoch))
        acc_mean.append(np.mean(acc_epoch))

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)
