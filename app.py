from flask import Flask, render_template, request , flash, redirect
from werkzeug.utils import secure_filename

import re

import torch
from torch import nn
from torchvision import models
from torchvision.transforms import v2
from torchvision.io import read_image
import torchvision

class_maps = {
    0: 'Abyssinian',
    1: 'american_bulldog',
    2: 'american_pit_bull_terrier',
    3: 'basset_hound',
    4: 'beagle',
    5: 'Bengal',
    6: 'Birman',
    7: 'Bombay',
    8: 'boxer',
    9: 'British_Shorthair',
    10: 'chihuahua',
    11: 'Egyptian_Mau',
    12: 'english_cocker_spaniel',
    13: 'english_setter',
    14: 'german_shorthaired',
    15: 'great_pyrenees',
    16: 'havanese',
    17: 'japanese_chin',
    18: 'keeshond',
    19: 'leonberger',
    20: 'Maine_Coon',
    21: 'miniature_pinscher',
    22: 'newfoundland',
    23: 'Persian',
    24: 'pomeranian',
    25: 'pug',
    26: 'Ragdoll',
    27: 'Russian_Blue',
    28: 'saint_bernard',
    29: 'samoyed',
    30: 'scottish_terrier',
    31: 'shiba_inu',
    32: 'Siamese',
    33: 'Sphynx',
    34: 'staffordshire_bull_terrier',
    35: 'wheaten_terrier',
    36: 'yorkshire_terrier'
 }

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet34()
num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(in_features = num_features, out_features = 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features = 256, out_features = 37)
)
model.load_state_dict(torch.load("./models/resnet34.pt"))
model.to(device)
model.eval()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

def is_file_allowed(filename):
    if re.findall(r"\.(.+)$", filename)[0] in ALLOWED_EXTENSIONS:
        return True

def file_extension(filename):
    return re.findall(r"\.(.+)$", filename)[0]

def classify(img):
    img = read_image(img, torchvision.io.image.ImageReadMode.RGB)
    img = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    
    output = model(img)
    _, pred = torch.max(output, 1)
    return pred.item()
    # return output
        
@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash("Request doesn't have a file part.")
            return redirect(request.url)
        
        file = request.files['file']
        
        # if the user doesn't select a file, the browser submits 
        # an empty file without a filename.
        if file.filename == "":
            flash("No file is selected.")
            return redirect(request.url)
        
        if file and is_file_allowed(file.filename):
            filename = secure_filename(file.filename)
            file.save("temp." + file_extension(filename))
            
        output = classify("temp." + file_extension(filename))
        result = class_maps[output]
        # result = output
        
        return render_template('index.html', result = result)
        
    return render_template('index.html')

