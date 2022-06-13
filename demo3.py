import pytorch_demo


model_path = r'E:\MiniProj\Image_segmentation\saved_model.pth'
model_ = UNet(num_classes = num_classes).to(device)
model_.load_state_dict(torch.load(model_path))