class Config:
    image_size = 256
    channels = 1

    z_dim = 256

    batch_size = 8
    epochs = 200

    lr_g = 1e-4
    lr_d = 2e-4

    T_max = 200
    T_min = 5
    sigma = 1.0  

    device = "cuda"