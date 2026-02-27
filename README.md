Dataset

Methods include
(1)Gene prioritization as a link prediction problem
(2)Network compiling
(3)Dimension Reduction for node raw representation
(4)RGCN Network Structure
(5)Negative Sampling

RGCN Network Structure:
class RGCN_Net(torch.nn.Module):
def __init__(self):
super(RGCN_Net, self).__init__()
self.conv1 = RGCNConv(rgcn_num_input,num_hidden1,
num_relations,num_bases=num_bases)
self.conv2 = RGCNConv(num_hidden1, out_channels,
num_relations,num_bases=num_bases)
self.bilinear = nn.Bilinear(out_channels,
out_channels,1,bias = False)

rgcn num input: 100,200,3000
num hidden1: 64
out channels: 2,4,8,16,32


The cross-entropy loss is used as the loss function, and only the edges
between disease nodes and gene nodes are considered.

Negative sampling: Randomly sample edges between disease nodes
and gene nodes that are not in the original graph
(1) We first set aside some negative edges for validation and test, of the
same size with positive validation and test edges
(2) Then in every iteration we will sample negative edges for training are
of the same size with the positive edges
(3) The model should be able to predict these random edges do not exist
and we reported the false positive rate
(4) One potential part to improve the model: neighborhood-based
negative sampling

Train set: 85%, validation set: 5%, test set: 10%
Full batch training
Adam optimizer with learning rate 0.001
Dropout rate 0.1
Three relations: gene-gene, disease-gene, disease-disease
Evaluation criteria: prediction accuracy, AUC (area under the ROC
curve), false positive rate, false negative rate
