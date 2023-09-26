import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy    

class ActionHeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60, num_joints=17, hidden_dim=2048, cam=False):
        super(ActionHeadClassification, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.cam = cam

        if self.cam:
            self.fcn = nn.Conv2d(dim_rep, num_classes, kernel_size=1)
        

    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        #print(feat.shape)
        N, M, T, J, C = feat.shape

        if self.cam:

            #print(feat.shape)
            feat = feat.permute (0,1,4,2,3)   # N, M, T, J, C -> (N, M), C, T, J
            feat = feat.view(-1, C, T, J)

            #print("Input to GAP: ", feat.shape)

            feat = F.avg_pool2d(feat, feat.shape[2:])
            feat = feat.view(N, M, -1, 1, 1).mean(dim=1)

            #print("Input to FCN: ", feat.shape)
            # prediction
            feat = self.fcn(feat)
            feat = feat.view(N, -1)

            #print(feat.shape)

            return feat, feat

        else: 
            feat = self.dropout(feat)
            feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
            feat = feat.mean(dim=-1)
            
            feat = feat.reshape(N, M, -1)  # (N, M, J*C)
            feat_weighted = feat.mean(dim=1)

            #print("feat2.shape: ", feat.shape)
            #feature1 = copy.deepcopy(feat)

            feat = self.fc1(feat_weighted)
            feat = self.bn(feat)

            
            feat = self.relu(feat)  

            my_feat = feat
            #print(feat.shape)
            #feature2 = copy.deepcopy(feat)
            #print("feat3.shape: ", feat.shape)

            feat = self.fc2(feat)
            return feat, my_feat, my_feat#, feature1, feature2

class ActionHeadClassification_p(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60, num_joints=17, hidden_dim=2048, cam=False):
        super(ActionHeadClassification_p, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc11 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc12 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.cam = cam

        

    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        #print(feat.shape)
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)

        feat_cp = feat

        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)  

        out_feat = self.fc2(feat)

        feat = self.fc11(feat_cp)
        feat = self.bn(feat)
        feat = self.relu(feat) 

        my_feat1 = feat 

        feat = self.fc12(feat_cp)
        feat = self.bn(feat)
        feat = self.relu(feat) 

        my_feat2 = feat 

        return out_feat, my_feat1, my_feat2#, feature1, feature2

class ActionHeadClassification_s(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60, num_joints=17, hidden_dim=2048, cam=False):
        super(ActionHeadClassification_s, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(int(hidden_dim/2), momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
        #print(hidden_dim, hidden_dim/2)
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc3 = nn.Linear(int(hidden_dim/2), num_classes)

        self.cam = cam

        

    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        #print(feat.shape)
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)

        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)  

        my_feat1 = feat

        feat = self.fc2(feat)
        feat = self.bn2(feat)
        feat = self.relu(feat)  

        my_feat2 = feat

        feat = self.fc3(feat)

        return feat, my_feat1, my_feat2#, feature1, feature2
    

        
class ActionHeadEmbed(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_joints=17, hidden_dim=2048):
        super(ActionHeadEmbed, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = F.normalize(feat, dim=-1)
        return feat

class ActionNet(nn.Module):
    def __init__(self, backbone, dim_rep=512, num_classes=60, dropout_ratio=0., \
                 version='class', hidden_dim=2048, num_joints=17, cam=False, arch= None, reshape_input = True):
        super(ActionNet, self).__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        self.reshape_input = reshape_input

        if version=='class':
            if arch == 'base':
                self.head = ActionHeadClassification(dropout_ratio=dropout_ratio, dim_rep=dim_rep, \
                                                     num_classes=num_classes, num_joints=num_joints, cam=cam)
            
            elif arch == "series":
                self.head = ActionHeadClassification_s(dropout_ratio=dropout_ratio, dim_rep=dim_rep, \
                                                     num_classes=num_classes, num_joints=num_joints, cam=cam)
            
            elif arch == "parallel":
                self.head = ActionHeadClassification_p(dropout_ratio=dropout_ratio, dim_rep=dim_rep, \
                                                     num_classes=num_classes, num_joints=num_joints, cam=cam)
                
            else:
                raise Exception('this head version is not implemented')
                
        elif version=='embed':
            self.head = ActionHeadEmbed(dropout_ratio=dropout_ratio, dim_rep=dim_rep, hidden_dim=hidden_dim, num_joints=num_joints)
        else:
            raise Exception('Version Error.')
        
    def forward(self, x):
        '''
            Input: (N, M x T x 17 x 3) 
        '''
        N, M, T, J, C = x.shape

        if self.reshape_input:
            x = x.reshape(N*M, T, J, C)        
        feat, attn_maps = self.backbone.get_representation(x)

        

        feat = feat.reshape([N, M, T, self.feat_J, -1])      # (N, M, T, J, C)

        attn_maps = attn_maps.reshape([N, M, T, self.feat_J, -1]) 

        attn_maps = torch.squeeze(attn_maps) # N, M, T, J

        out, my_feat1, my_feat2 = self.head(feat)

        return out, my_feat1, my_feat2, attn_maps# my_feat.permute(0,4,2,3,1) #attn_maps#,  # N, C, T, J, M


class MaskCLR(nn.Module):
    def __init__(self, backbone, dim_rep=512, num_classes=60, dropout_ratio=0., \
                 version='class', hidden_dim=2048, num_joints=17, cam=False, arch= None, mask_th=0.2):
        super(MaskCLR, self).__init__()
        
        self.n_branches = 2

        self.mask_th = mask_th
        self.model = ActionNet(backbone=backbone, dim_rep=dim_rep, num_classes=num_classes,\
                       dropout_ratio=dropout_ratio, version=version, hidden_dim=hidden_dim, num_joints=num_joints, \
                        cam=cam, arch= arch)

    def get_mask(self, branch_attn): #weight, feature
        N, M, T, J = branch_attn.shape
        branch_attn = branch_attn.view(-1)

        branch_attn[branch_attn < 0] = 0
        branch_attn = branch_attn / torch.max(branch_attn)


        branch_attn[branch_attn > self.mask_th] = 1
        branch_attn[branch_attn <= self.mask_th] = 0
        branch_attn = 1 - branch_attn
        mask = branch_attn.view(N, M, T, J)

        mask = mask.unsqueeze(4).repeat(1, 1, 1, 1, 3)


        return mask

    def forward(self, inp, inp_n):
        '''
            Input: (N, M x T x 17 x 3) 
        '''

        out = []
        feature1 = []
        feature2 = []
        j_importances = []
        masked_inps = []

        N, M, T, J, C = inp.shape
        mask = torch.ones(N, M, T, J, C).cuda()

        x = inp

        for idx in range(self.n_branches):
            
            x = x * mask if (idx < self.n_branches - 1) else inp_n # x * (1-mask)#

            temp_out, temp_feature1, temp_feature2, j_importance = self.model(x)
            mask = self.get_mask(j_importance)

            # output
            #print(torch.unique(mask))

            masked_inps.append(x)
            out.append(temp_out) #.unsqueeze(-1)
            feature1.append(temp_feature1)
            feature2.append(temp_feature2)
            j_importances.append(j_importance)
        
        

        return out, feature1, feature2, j_importances, masked_inps