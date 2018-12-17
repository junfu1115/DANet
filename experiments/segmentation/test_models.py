import importlib
import torch
import encoding
from option import Options
from torch.autograd import Variable

if __name__ == "__main__":
    args = Options().parse()
    model = encoding.models.get_segmentation_model(args.model, dataset=args.dataset, aux=args.aux,
                                                   backbone=args.backbone,
                                                   se_loss=args.se_loss, norm_layer=torch.nn.BatchNorm2d)
    print('Creating the model:')
    
    print(model)
    model.cuda()
    model.eval()
    x = Variable(torch.Tensor(4, 3, 480, 480)).cuda()
    with torch.no_grad():
        out = model(x)
    for y in out:
        print(y.size())
