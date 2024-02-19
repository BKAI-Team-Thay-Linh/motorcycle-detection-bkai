import os
import motorbike_project as mp
import torch
import argparse

from PIL import Image
import polars as pl

torch.multiprocessing.set_sharing_strategy('file_system') # This is important for Windows users

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='resnet18', help='model name')
parser.add_argument('--checkpoint', '-c', type=str, help='path to the checkpoint of the model')
parser.add_argument('--bbox', '-b', type=str, help='path to the bounding box to be infered')
parser.add_argument('--save_path', '-sp', type=str, help='path to save the output (csv file)')

args = parser.parse_args()

def predict(args):
    model = mp.MotorBikeModel(
        mode='infer',
        model=args.model,
        num_classes=5,
        weight_path=args.checkpoint
    )
    
    output = []
    
    for img in os.listdir(args.bbox):
        item = {
            'BBox Name': img,
            'Predicted Class': ''
        }
        img_path = os.path.join(args.bbox, img)
        img = Image.open(img_path)

        pred = model.infer(img)
        
        item['Predicted Class'] = pred.item()

        print(f'Image: {img_path}, Predicted class: {pred.item()}')
        
    # Save the output to a csv file
    df = pl.DataFrame(output)
    df.write_csv(args.save_path)
    
    print('Output saved to:', args.save_path)

if __name__=='__main__':
    predict(args)