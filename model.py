import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

# Initialize CNN structure
def structure_init(name):
    if name == 'pretrain_resnet50':
        resnet50 = models.resnet50(pretrained=True)
        structure = resnet50
        out_features = structure.fc.out_features            
    elif name == 'pretrain_resnet152':
		resnet152 = models.resnet152(pretrained=True)
		structure = resnet152
		out_features = structure.fc.out_features
    elif name == 'pretrain_densenet121':
		densenet121 = models.densenet121(pretrained=True)
		structure = densenet121
		out_features = structure.classifier.out_features		
    elif name == 'pretrain_densenet121_freeze':
		densenet121 = models.densenet121(pretrained=True)		
		structure = densenet121
		out_features = structure.classifier.out_features
		
	# if want freeze pretrain params
	# for param in densenet121.parameters():
	#		  param.requires_grad = False         
      
    fully_connect = nn.Sequential(
		nn.Linear(out_features, 4096),
		nn.ReLU(),
		nn.Dropout(),
		nn.Linear(4096, 4096),
		nn.ReLU(),
		nn.Dropout(),
		nn.Linear(4096, 1),
		nn.Sigmoid(),
	)
    return structure, fully_connect

class SiameseNetwork(nn.Module):
    def __init__(self, structure ='simple'):
		super(SiameseNetwork, self).__init__()            
		self.feature_extract, self.fully_connect = structure_init(structure)
		
    def forward(self, input1, input2):
		output1 = self.feature_extract(input1)
		output2 = self.feature_extract(input2)            

		# L1 norm
		# dis = torch.abs(output1 - output2)		
		# L2 norm
		dis = torch.sqrt(torch.pow(output1 - output2, 2) + 1e-8)
		
		dis = torch.flatten(dis, 1)
		out = self.fully_connect(dis)
        return out

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, predict, label):
        loss_contrastive = torch.mean((1-label) * torch.pow(predict, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - predict, min=0.0), 2))
 
        return loss_contrastive

# For test
if __name__ == '__main__':
      net = SiameseNetwork('pretrain_resnet50')
      print(net)      
      # summary(SiameseNetwork('pretrain_resnet50').to('cuda'), [(1, 120, 120), (1, 120, 120)])