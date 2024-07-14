# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# command for train
python train.py flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu
# command for predict
python predict.py flowers/test/1/image_06743.jpg checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu