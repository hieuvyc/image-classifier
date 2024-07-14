#author:Hieu Tran
import argparse
import torch
from torchvision import models
from model import load_checkpoint
from utils import process_image, load_category_names

def predict(image_path, checkpoint, top_k, category_names, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(checkpoint)
    model.to(device)
    
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(top_k, dim=1)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class.cpu().numpy()[0]]
    
    if category_names:
        cat_to_name = load_category_names(category_names)
        top_class = [cat_to_name[str(cls)] for cls in top_class]
    
    return top_p.cpu().numpy()[0], top_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category to names mapping JSON')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()
    
    probs, classes = predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)
    for i in range(len(classes)):
        print(f"{classes[i]}: {probs[i]*100:.2f}%")
