'training' : transforms.Compose([
        transforms.CenterCrop(500),
        transforms.FiveCrop((200,250))
        transforms.Grayscale(num_output_channels=1)
        transforms.ToTensor(),
        transforms.Normalize([0.4363,0.4328,0.3291],[0.2132,0.2078,0.2040])
    ]),
