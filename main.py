import os
import h5py
import numpy as np
import scipy.sparse as sp
from funs import my_preprocess


n_genes = 12331
n_dis = 3215

##gene interation network
gene_phenes_path = './data_prioritization/genes_phenes.mat'
f = h5py.File(gene_phenes_path, 'r')
gene_network_adj = sp.csc_matrix((np.array(f['GeneGene_Hs']['data']),
    np.array(f['GeneGene_Hs']['ir']), np.array(f['GeneGene_Hs']['jc'])),
    shape=(n_genes,n_genes))
gene_network_adj = gene_network_adj.tocsr()


#find edge list from nonzero interations
#gene network is symmetric
gene_network_arr = gene_network_adj.toarray()
gene_edge_arr = np.argwhere(gene_network_arr!=0)
#gene_network has no self edge
#np.where(gene_edge_arr[:,0]==gene_edge_arr[:,1])


##disease similarity network
disease_network_adj = sp.csc_matrix((np.array(f['PhenotypeSimilarities']['data']),
    np.array(f['PhenotypeSimilarities']['ir']), np.array(f['PhenotypeSimilarities']['jc'])),
    shape=(n_dis, n_dis))
disease_network_adj = disease_network_adj.tocsr()
disease_network_adj = my_preprocess.network_edge_threshold(disease_network_adj, 0.2)


#find edge list from nonzero interations
#disease network is symmetric dis_network_arr[2,730] = dis_network_arr[730,2]
dis_network_arr = disease_network_adj.toarray()
dis_edge_arr = np.argwhere(dis_network_arr!=0)
##disease network have self edges,remove self edge
mask = (dis_edge_arr[:,0]!=dis_edge_arr[:,1])
dis_edge_arr = dis_edge_arr[mask,:]

##gene and disease association network
dg_ref = f['GenePhene'][0][0]
gene_disease_adj = sp.csc_matrix((np.array(f[dg_ref]['data']),
    np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
    shape=(12331, 3215))
gene_disease_adj = gene_disease_adj.tocsr()

#find edge list from nonzero interations
#gene_dis network is not symmetric: gene_dis_arr[2,939] != gene_dis_arr[939,2]
gene_dis_arr = gene_disease_adj.toarray()
gene_dis_edge_arr = np.argwhere(gene_dis_arr!=0)

##find nonexiting edge indices
gene_dis_noedge_arr = np.argwhere(gene_dis_arr==0)


#####################construct node features############
##gene node features
gene_feature_path = './data_prioritization/GeneFeatures.mat'
f_gene_feature = h5py.File(gene_feature_path,'r')
gene_feature_exp = np.array(f_gene_feature['GeneFeatures'])
gene_feature_exp = np.transpose(gene_feature_exp)

###conduct pca on gene expression microarray data
pca_model = my_preprocess.my_pca(n_components=100)
gene_microarray_pca = pca_model.fit_transform(gene_feature_exp)

##gene-phenotype associations of other species
row_list = [3215, 1137, 744, 2503, 1143, 324, 1188, 4662, 1243]
gene_feature_list_other_spe = list()
for i in range(1,9):
    dg_ref = f['GenePhene'][i][0]
    disease_gene_adj_tmp = sp.csc_matrix((np.array(f[dg_ref]['data']),
        np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
        shape=(12331, row_list[i]))
    gene_feature_list_other_spe.append(disease_gene_adj_tmp)

##combine gene expression microarray data with gene-phenotype associations of other species
gene_feat = sp.hstack(gene_feature_list_other_spe+[gene_microarray_pca])
gene_feat = sp.csc_matrix(gene_feat)
pca_model2 = my_preprocess.my_pca(n_components=3000)
gene_feat_full_pca = pca_model2.fit_transform(gene_feat.toarray())

##disease features
disease_tfidf_path = './data_prioritization/clinicalfeatures_tfidf.mat'
f_disease_tfidf = h5py.File(disease_tfidf_path)
disease_tfidf = np.array(f_disease_tfidf['F'])
disease_tfidf = np.transpose(disease_tfidf)
disease_tfidf = sp.csc_matrix(disease_tfidf)


###conduct pca on disease feature data
dis_feat_arr = disease_tfidf.toarray()
dis_feat_pca = pca_model2.fit_transform(dis_feat_arr)

##save network and node feature data to file
"""
np.save('gene_feat_full_pca.npy', gene_feat_full_pca)
np.save('dis_feat_pca.npy', dis_feat_pca)
np.save('gene_edge_arr.npy', gene_edge_arr)
np.save('dis_edge_arr.npy', dis_edge_arr)
np.save('gene_dis_edge_arr.npy', gene_dis_edge_arr)
np.save('gene_dis_noedge_arr.npy', gene_dis_noedge_arr)
"""

###############construct node feature for graph
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
#x is node feature,[num.nodes,num.node.features]
feat_full_arr = np.concatenate((gene_feat_full_pca, dis_feat_pca), axis=0)
x = torch.tensor(feat_full_arr , dtype=torch.float)

gene_node_type = [0]*n_genes
dis_node_type = [1]*n_dis
#y is node type,one dimensional list
y = torch.tensor(gene_node_type+dis_node_type, dtype=torch.float)

###############construct edge index for graph#################
source_gene_nodes = gene_edge_arr[:,0]
target_gene_nodes = gene_edge_arr[:,1]
#adjust node indices of disease nodes
source_dis_nodes = dis_edge_arr[:,0]+n_genes
target_dis_nodes = dis_edge_arr[:,1]+n_genes

source_gene_dis_nodes = gene_dis_edge_arr[:,0]
target_gene_dis_nodes = gene_dis_edge_arr[:,1]+n_genes

source_nodes = np.concatenate((source_gene_nodes,source_gene_dis_nodes,target_gene_dis_nodes,source_dis_nodes),axis = 0)
target_nodes = np.concatenate((target_gene_nodes,target_gene_dis_nodes,source_gene_dis_nodes,target_dis_nodes),axis = 0)

edge_arr = np.stack((source_nodes,target_nodes))
edge_index = torch.tensor(edge_arr, dtype=torch.long)

###############construct edge type for graph#################
gene_edge_type = [0]*len(source_gene_nodes)
gene_dis_edge_type = [1]*len(source_gene_dis_nodes)
dis_gene_edge_type = [1]*len(source_gene_dis_nodes)
dis_edge_type = [2]*len(source_dis_nodes)
edge_type = torch.tensor(gene_edge_type  + gene_dis_edge_type  + dis_gene_edge_type + dis_edge_type,dtype = torch.long)

data = Data(x=x, y=y, edge_index=edge_index,edge_type = edge_type)

##############choose train and test edges###############
val_ratio = 0.05
test_ratio = 0.1
row, col = data.edge_index

pos_sample_edge_range = torch.tensor(list(range(row.size(0))))
mask = row < col
urow, ucol = row[mask], col[mask]
# Return upper triangular portion
pos_sample_edge_range = pos_sample_edge_range[mask]
n_v = int(np.floor(val_ratio * data.num_edges*0.5))
n_t = int(np.floor(test_ratio * data.num_edges*0.5))

pos_sample_edge_range_arr = pos_sample_edge_range.numpy()
np.random.shuffle(pos_sample_edge_range_arr)

pos_val_idx = torch.tensor(pos_sample_edge_range_arr[:n_v])
pos_test_idx = torch.tensor(pos_sample_edge_range_arr[n_v:n_v+n_t])
pos_train_idx = torch.tensor(pos_sample_edge_range_arr[n_v+n_t:])

# Positive edges for all validation and test
data.val_pos_edge_index = torch.stack([row[pos_val_idx], col[pos_val_idx]], dim=0)
data.test_pos_edge_index = torch.stack([row[pos_test_idx], col[pos_test_idx]], dim=0)
##single direction for train
train_pos_edge_index = torch.stack([row[pos_train_idx], col[pos_train_idx]], dim=0)

##filter out gene-disease interaction for validation and test
heter_pos_val_idx = pos_val_idx[edge_type[pos_val_idx]==1]
heter_pos_test_idx = pos_test_idx[edge_type[pos_test_idx]==1]
data.val_heter_pos_edge_index = torch.stack([row[heter_pos_val_idx], col[heter_pos_val_idx]], dim=0)
data.test_heter_pos_edge_index = torch.stack([row[heter_pos_test_idx], col[heter_pos_test_idx]], dim=0)

##create duplicates of postive train in opposite direction
data.train_pos_edge_index = to_undirected(train_pos_edge_index)

##filter out gene-disease interaction for train
heter_pos_train_idx = pos_train_idx[edge_type[pos_train_idx]==1]
train_heter_pos_edge_index = torch.stack([row[heter_pos_train_idx], col[heter_pos_train_idx]], dim=0)
data.train_heter_pos_edge_index = to_undirected(train_heter_pos_edge_index)

##find indices of opposite directions of train to extract edge types
x0 = train_pos_edge_index
x1 = torch.stack([x0[1],x0[0]],dim = 0)
x2 = torch.transpose(x1, 0, 1)
x2_arr = x2.numpy()
edge_arr = torch.transpose(edge_index,0,1).numpy()
result = my_preprocess.inNd(edge_arr,x2_arr)
transpose_pos_train_edge_ind = np.where(result==True)[0]
data.train_pos_edge_type = torch.cat((edge_type[pos_train_idx],edge_type[transpose_pos_train_edge_ind]),0)

#############  Select negative edges for validation and test
neg_source_gene_dis_nodes = gene_dis_noedge_arr[:,0] #gene_dis_noedge_arr:(39640211, 2)
neg_target_gene_dis_nodes = gene_dis_noedge_arr[:,1]+n_genes
neg_edge_idx = list(range(neg_source_gene_dis_nodes.shape[0]))
np.random.shuffle(neg_edge_idx)
n_neg_v = heter_pos_val_idx.size(0)
n_neg_t = heter_pos_test_idx.size(0)
neg_val_idx = torch.tensor(neg_edge_idx[:n_neg_v],dtype = torch.long)
neg_test_idx = torch.tensor(neg_edge_idx[n_neg_v:n_neg_v+n_neg_t],dtype = torch.long)

neg_val_row = torch.tensor(neg_source_gene_dis_nodes[neg_val_idx])
neg_val_col = torch.tensor(neg_target_gene_dis_nodes[neg_val_idx])
data.val_neg_edge_index = torch.stack([neg_val_row, neg_val_col], dim=0)

neg_test_row = torch.tensor(neg_source_gene_dis_nodes[neg_test_idx])
neg_test_col = torch.tensor(neg_target_gene_dis_nodes[neg_test_idx])
data.test_neg_edge_index = torch.stack([neg_test_row, neg_test_col], dim=0)

#remove selected test and validation from negative sampling for training data
delete_for_neg_train = torch.cat((neg_val_idx,neg_test_idx),0)
neg_train_sample_base = np.delete(gene_dis_noedge_arr,delete_for_neg_train,0) #size(39639572, 2)

#############  Negative Sampling Function
def my_negative_sampling(neg_train_sample_base, n_genes,num_neg_samples=100):
    #Samples random negative edges
    neg_source_gene_dis_nodes = neg_train_sample_base[:,0]
    neg_target_gene_dis_nodes = neg_train_sample_base[:,1]+n_genes
    neg_edge_idx = list(range(neg_source_gene_dis_nodes.shape[0]))
    np.random.shuffle(neg_edge_idx)
    neg_test_idx = torch.tensor(neg_edge_idx[:num_neg_samples],dtype = torch.long)

    neg_test_row = torch.tensor(neg_source_gene_dis_nodes[neg_test_idx])
    neg_test_col = torch.tensor(neg_target_gene_dis_nodes[neg_test_idx])
    neg_edge_index = torch.stack([neg_test_row, neg_test_col], dim=0).long()
    return neg_edge_index

############## Start Learning##########################
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, RGCNConv
import torch.nn as nn

# The relational graph convolutional operator from the `"ModelingRelational Data with Graph Convolutional Networks"
class RGCN_Net(torch.nn.Module):
    def __init__(self):
        super(RGCN_Net, self).__init__()
        self.conv1 = RGCNConv(rgcn_num_input, num_hidden1, num_relations,
                              num_bases=num_bases)
        self.conv2 = RGCNConv(num_hidden1, out_channels, num_relations,
                              num_bases=num_bases)
        self.bilinear = nn.Bilinear(out_channels, out_channels, 1,bias = False)

    def encode(self):
        x = F.relu(self.conv1(rgcn_x, data.train_pos_edge_index, data.train_pos_edge_type))
        x = F.dropout(x, p = dropout_rate)
        x = self.conv2(x, data.edge_index, data.edge_type)
        if(act_flag=="softmax"):
           x = F.softmax(x, dim=1)
        if(act_flag=="log_softmax"):
           x = F.log_softmax(x, dim=1)
        return x

    def inner_decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        logits = logits.sigmoid()
        return logits
    
    def bilinear_decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        input1 = z[edge_index[0]]
        input2 = z[edge_index[1]]
        logits = self.bilinear(input1, input2).reshape(-1)
        logits = logits.sigmoid()
        return logits

#output size
num_hidden1 = 64
out_channels = 4 #could change to 2,4,8,16
#number of relations
num_relations = 3
#regularization parameter
#num_bases = None
num_bases = 1
#activation funtion flag
act_flag = "softmax"
learning_rate = 0.001
dropout_rate = 0.1
##input size and input raw features
rgcn_num_input,rgcn_x = data.num_node_features, data.x
link_label_th = 0.5 #prediction accuracy threshold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RGCN_Net()
model, data = model.to(device), data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train():
    model.train()
    pos_size = data.train_heter_pos_edge_index.size(1)
    neg_edge_index = my_negative_sampling(neg_train_sample_base, n_genes, num_neg_samples=pos_size)
    optimizer.zero_grad()
    z = model.encode()
    link_logits = model.bilinear_decode(z, data.train_heter_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_heter_pos_edge_index, neg_edge_index)
    link_masks = link_logits>=link_label_th
    pred_labels = torch.zeros(pos_size*2)
    pred_labels[link_masks] = 1
    loss = F.binary_cross_entropy(link_logits, link_labels,reduction='sum')
    perf = roc_auc_score(link_labels.cpu(), link_logits.detach().numpy())
    acc = roc_auc_score(link_labels.cpu(), pred_labels.detach().numpy())
    false_neg = torch.where(pred_labels[0:pos_size]==0)[0].size(0)/pos_size
    false_pos = torch.where(pred_labels[pos_size:]==1)[0].size(0)/pos_size
    loss.backward()
    optimizer.step()
    return loss,perf,acc,false_neg,false_pos

@torch.no_grad()
def test():
    model.eval()
    perfs = []
    losses = []
    accs = []
    false_negs = []
    false_poses = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_heter_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode()
        link_logits = model.bilinear_decode(z, pos_edge_index, neg_edge_index)
        link_masks = link_logits>=link_label_th
        pos_size = pos_edge_index.size(1)
        pred_labels = torch.zeros(pos_size*2)
        pred_labels[link_masks] = 1
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        loss = F.binary_cross_entropy(link_logits, link_labels,reduction='sum')
        losses.append(loss)
        perfs.append(roc_auc_score(link_labels.cpu(), link_logits.cpu()))
        accs.append(roc_auc_score(link_labels.cpu(), pred_labels))
        false_negs.append(torch.where(pred_labels[0:pos_size]==0)[0].size(0)/pos_size)
        false_poses.append(torch.where(pred_labels[pos_size:]==1)[0].size(0)/pos_size)
    return losses,perfs,accs,false_negs,false_poses

n_epoch = 100
best_val_perf = test_perf = 0
loss_arr = np.full((n_epoch-1,3), np.inf)
auc_arr = np.empty([n_epoch-1,3])
acc_arr = np.empty([n_epoch-1,3])
false_neg_arr = np.empty([n_epoch-1,3])
false_pos_arr = np.empty([n_epoch-1,3])
for epoch in range(1, n_epoch):
    train_loss,train_perf,train_acc,train_false_neg,train_false_pos = train()
    #val_perf, tmp_test_perf = test()
    ls_loss, ls_perf,ls_acc,ls_false_neg,ls_false_pos = test()
    val_perf, tmp_test_perf = ls_perf[0],ls_perf[1]
    val_loss, test_loss = ls_loss[0],ls_loss[1]
    val_acc, tmp_test_acc = ls_acc[0],ls_acc[1]
    """
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf
    """
    loss_arr[epoch-1,:] = np.array([train_loss,val_loss,test_loss])
    auc_arr[epoch-1,:] = np.array([train_perf,val_perf,tmp_test_perf])
    acc_arr[epoch-1,:] = np.array([train_acc,val_acc,tmp_test_acc])
    false_neg_arr[epoch-1,:] = np.array([train_false_neg,ls_false_neg[0],ls_false_neg[1]])
    false_pos_arr[epoch-1,:] = np.array([train_false_pos,ls_false_pos[0],ls_false_pos[1]])
    log = 'Epoch: {:03d}, TrLoss: {:.4f}, TrPerf: {:.4f}, TrAcc: {:.4f}| VLoss: {:.4f}, VPerf: {:.4f}, VAcc: {:.4f}| TLoss: {:.4f}, TPerf: {:.4f}, TAcc: {:.4f}'
    print(log.format(
        epoch, train_loss, train_perf, train_acc, 
        val_loss, val_perf, val_acc, 
        test_loss, tmp_test_perf, tmp_test_acc))


loss_arr = np.around(loss_arr,decimals = 4)
auc_arr = np.around(auc_arr,decimals = 4)
acc_arr = np.around(acc_arr,decimals = 4)
false_neg_arr = np.around(false_neg_arr,decimals = 4)
false_pos_arr = np.around(false_pos_arr,decimals = 4)
n_epoch = n_epoch-1
np.save(f'act_flag_{act_flag}_n_epoch-{n_epoch}_out_channels-{out_channels}_loss_arr',loss_arr)
np.save(f'act_flag_{act_flag}_n_epoch-{n_epoch}_out_channels-{out_channels}_auc_arr',auc_arr)
np.save(f'act_flag_{act_flag}_n_epoch-{n_epoch}_out_channels-{out_channels}_acc_arr',acc_arr)
np.save(f'act_flag_{act_flag}_n_epoch-{n_epoch}_out_channels-{out_channels}_false_neg_arr',false_neg_arr)
np.save(f'act_flag_{act_flag}_n_epoch-{n_epoch}_out_channels-{out_channels}_false_pos_arr',false_pos_arr)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
def save_plots(results, s, out_channels, activation):
    """Plot

        Plot two figures: loss vs. epoch and accuracy vs. epoch
    """
    n = len(results)
    xs = np.arange(n)
    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, results[:,0], ':', linewidth=2, label='Train')
    ax.plot(xs, results[:,1], '--', linewidth=2, label='Valid')
    ax.plot(xs, results[:,2], '-', linewidth=2, label='Test')
    ax.set_xlabel("Epoch")
    ax.set_ylabel(s)
    ax.legend(loc='upper left')
    plt.title('out_channels = ' + str(out_channels) + ', ' + activation)
    plt.savefig(activation + '_full_n_epoch'+ str(n) + '_out_channels_' +str(out_channels) + '_' + s)

save_plots(auc_arr, 'AUC', out_channels, act_flag)
save_plots(acc_arr, 'Pred_acc', out_channels, act_flag)
save_plots(false_neg_arr, 'False_negative', out_channels, act_flag)
save_plots(false_pos_arr, 'False_positive', out_channels, act_flag)