import torch
import torch.nn.functional as F
import numpy as np


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    def __init__(self):

      super().__init__()

      layers = []
      layers.append(torch.nn.Conv2d(3,16,3,1,0)) # padding=0 to capture the edges of images with the 3x3 filter
      layers.append(torch.nn.BatchNorm2d(16)) # standardize inputs of the model after each convolutional layer to converge faster in training
      layers.append(torch.nn.ReLU()) 
      layers.append(torch.nn.Conv2d(16,32, 5,1,0))
      layers.append(torch.nn.BatchNorm2d(32))
      layers.append(torch.nn.ReLU()) # MODEL ARCH: input image, convolutional layer, batchnorm2d, nonlinearity, pooling
      layers.append(torch.nn.Conv2d(32,32,7,2,0))
      layers.append(torch.nn.BatchNorm2d(32))
      layers.append(torch.nn.ReLU())
      layers.append(torch.nn.Conv2d(32,16 ,3,1,0))
      layers.append(torch.nn.BatchNorm2d(16))
      layers.append(torch.nn.ReLU())
      layers.append(torch.nn.MaxPool2d(3,1,0))
      layers.append(torch.nn.MaxPool2d(3,1,0))
      layers.append(torch.nn.MaxPool2d(3,1,0)) # small kernels instead of fully connected network allows better weight sharing, faster training/testing
      #dropout?? nn.Dropout2d(0.25)
      # nn.Dropout2d(0.5)
      layers.append(torch.nn.Linear(50,128)) # fully connected layer 1, default bias=True, this learns its own bias
      layers.append(torch.nn.ReLU())
      layers.append(torch.nn.Linear(128,2)) # applies a linear transformation y=xAT + b
      self._conv = torch.nn.Sequential(*layers)


    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
       #img = color.rgb2gray(img)
        #img = lit.rgb2gray_approx(img)
        #print(img.shape)
        #color_img = np.asarray(img) / 255
        #img = np.mean(img, axis=2)
        #print(img.shape)
        x = self._conv(img)
        #x *= torch.sigmoid(x) # swish activation
        #x = self.pool(x)
        #x = self.bn1(x)
        #x = x.view(-1, 1) # flatten
        #x = F.log_softmax(self.lin1(x), dim=1) # softmax, fully connected
        #x *= torch.sigmoid(x)
        #x = F.relu(x)
        #x = F.relu(self.lin1(x))
        #x = F.relu(self.lin2(x))
        #x = F.relu(self.lin1(x))
        #print(img.shape)
        #print(x.shape)
        #return x
        return spatial_argmax(x[:, 0])
       # return self.classifier(x.mean(dim=[-2, -1]))


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from controller import control
    from utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
