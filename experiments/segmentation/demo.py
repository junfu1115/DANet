import torch
import encoding

# Get the model
model = encoding.models.get_model('fcn_resnet50s_ade', pretrained=True).cuda()
model.eval()

# Prepare the image
url = 'https://github.com/zhanghang1989/image-data/blob/master/' + \
      'encoding/segmentation/ade20k/ADE_val_00001142.jpg?raw=true'
filename = 'example.jpg'
img = encoding.utils.load_image(
    encoding.utils.download(url, filename)).cuda().unsqueeze(0)

# Make prediction
output = model.evaluate(img)
predict = torch.max(output, 1)[1].cpu().numpy() + 1

# Get color pallete for visualization
mask = encoding.utils.get_mask_pallete(predict, 'ade20k')
mask.save('output.png')
