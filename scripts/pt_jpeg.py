
import torch
import torchvision.transforms as transforms

# Load the Torch file
torch_file = torch.load('/mnt/sda/abka03-data/mcx/77_nonrobust/orginal_image/image_index_8868.pt')
# Convert the tensor to a PIL image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Resize the image to desired dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])
normalized_image = transform(torch_file)

# Convert the normalized image tensor back to a PIL image
pil_image = transforms.ToPILImage()(normalized_image)


# Save the PIL image as a JPEG file
pil_image.save('/home/abka03/IML/xai-backdoors/imgs/sample.jpg', 'JPEG')