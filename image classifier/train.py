#author:Hieu Tran
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import json
from model import build_model, save_checkpoint
from utils import load_data

def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    dataloaders, class_to_idx = load_data(data_dir)
    
    model = build_model(arch, hidden_units, len(class_to_idx))
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    steps = 0
    running_loss = 0
    print_every = 40
    
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        validation_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                
                running_loss = 0
                model.train()
    
    save_checkpoint(model, save_dir, arch, hidden_units, class_to_idx, optimizer, epochs)
    print("Model trained and checkpoint saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')
    parser.add_argument('data_dir', type=str, help='Directory containing dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16, alexnet, etc.)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()
    
    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
