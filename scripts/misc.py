

if __name__ == "__main__":

    path = '../input/understanding_cloud_organization'
    os.listdir(path)

    train = pd.read_csv(f'{path}/train.csv')
    sub = pd.read_csv(f'{path}/sample_submission.csv')
    
    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
    reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda'

    ACTIVATION = None
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    num_workers = 0
    bs = 16
    train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms = get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    num_epochs = 21
    logdir = "./logs/segmentation"

    # model, criterion, optimizer
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    runner = SupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )
    utils.plot_metrics(
        logdir=logdir,
        # specify which metrics we want to plot
        metrics=["loss", "dice", 'lr', '_base/lr']
    )
