import argparse
from tqdm import tqdm
from torch.utils import data
import torchvision.transforms as transform
from encoding.datasets import get_segmentation_dataset

def main():
    parser = argparse.ArgumentParser(description='Test Dataset.')
    parser.add_argument('--dataset', type=str, default='ade20k',
                        help='dataset name (default: pascal12)')
    args = parser.parse_args()

    input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    trainset = get_segmentation_dataset(args.dataset, split='val', mode='train',
                                        transform=input_transform)
    trainloader = data.DataLoader(trainset, batch_size=16,
                                  drop_last=True, shuffle=True)
    tbar = tqdm(trainloader)
    max_label = -10
    for i, (image, target) in enumerate(tbar):
        tmax = target.max().item()
        tmin = target.min().item()
        assert(tmin >= -1)
        if tmax > max_label:
            max_label = tmax
        assert(max_label < trainset.NUM_CLASS)
        tbar.set_description("Batch %d, max label %d"%(i, max_label))

if __name__ == "__main__":
    main()
