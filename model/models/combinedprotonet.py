import torch
import torch.nn as nn
from model.models import FewShotModel
from model.models.mutilattention import MutilheadAttention
from model.models.Relationnet import RelationNetwork
import math


def euclidean_distance(feats, prototypes):
    # Convert the tensors in a shape that is easily computable
    n = feats.size(0)
    m = prototypes.size(0)
    d = feats.size(2)

    if d != prototypes.size(1):
        raise ValueError("Features and prototypes are of different size")

    prototypes = prototypes.unsqueeze(0).expand(n, m, d)
    return torch.pow(feats - prototypes, 2).sum(dim=2)


class CombinedProtoNet(FewShotModel):
    def __init__(self, args, sem_feat_dim=300):
        super().__init__(args)
        if args.backbone_class == "ConvNet":
            hdim = 64
        elif args.backbone_class == "Res12":
            hdim = 640
        elif args.backbone_class == "Res18":
            hdim = 512
        elif args.backbone_class == "WRN":
            hdim = 640
        else:
            raise ValueError("Unknown Backbone %s" % args.backbone_class)


        self.args = args
        self.sem_feat_dim = sem_feat_dim
        self.hidden_dim = hdim
        self.sp_attention = MutilheadAttention(hid_dim=640, n_heads=8, dropout=0.4)
        ##################################################################
        self.ch_attention = nn.Sequential(nn.Linear(sem_feat_dim, 150), nn.LeakyReLU(0.1),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(150, hdim), nn.Sigmoid())
        self.MLP2 =  nn.Sequential(nn.Linear(sem_feat_dim, 150), nn.LeakyReLU(0.1), \
                                       nn.Dropout(0.1), \
                                       nn.Linear(150,hdim), nn.Sigmoid())
        self.relation_network1=RelationNetwork(hdim*2,64)
        self.relation_network2 = RelationNetwork(hdim+sem_feat_dim, 32)
        ###################################################################
        self.encode_vis = nn.Sequential(nn.Linear(hdim, 32),
                                        nn.Dropout(p=0.2),
                                        nn.ReLU(),
                                        nn.Linear(32, 32))
        self.encode_sem = nn.Sequential(nn.Linear(sem_feat_dim, 32),
                                        nn.Dropout(p=0.6),
                                        nn.ReLU(),
                                        nn.Linear(32, 32))
        self.act = nn.ReLU()
        self.pred_hadamard = nn.Sequential(nn.Linear(sem_feat_dim, 32),
                                           nn.Softmax(dim=1),
                                           nn.Linear(32, hdim)
                                           )
        self.g_linear1 = nn.Linear(sem_feat_dim, sem_feat_dim)
        self.dropout_g = nn.Dropout(p=0.4)
        self.g_linear2 = nn.Linear(sem_feat_dim, hdim)
        ##################################################################

    def _forward(self, instance_embs, attrib, support_idx, query_idx):
        args = self.args
        b,c,w,h=instance_embs.shape
        e_attrib = attrib.repeat_interleave(args.shot, dim=0)
        support = instance_embs.view(args.way, args.shot + args.query, c,h,w)[:, :args.shot, :].contiguous().view(
            args.way * args.shot, c,h,w)
        query = instance_embs.view(args.way, args.shot + args.query, c,h,w)[:, args.shot:, :].contiguous().view(
            args.way * args.query, c,h,w)
        attrib = self.MLP2(e_attrib)
        attrib_s_1=attrib.unsqueeze(2)
        attrib_s=torch.repeat_interleave(attrib_s_1,h*w,dim=2).reshape(args.way*args.shot,c,h,w)
        attrib_s_2=e_attrib.unsqueeze(2)
        attrib_s_2=torch.repeat_interleave(attrib_s_2,h*w,dim=2).reshape(args.way*args.shot,self.sem_feat_dim,h,w)
        #################################################
        attrib_q=attrib_s_1.repeat((args.way*args.query,1,1))
        attrib_q=torch.repeat_interleave(attrib_q,h*w,dim=2).reshape(args.way*args.query*args.way,c,h,w)
        query_ins=torch.repeat_interleave(query,args.way,dim=0)
        ########################################
        support_sp=self.sp_attention(support,attrib_s,support)
        c_attention = self.ch_attention(e_attrib)
        c_attention = c_attention.unsqueeze(2)
        c_attention_s = torch.repeat_interleave(c_attention, h*w, dim=2).view(args.way*args.shot, c, h, w)
        support_feats=support_sp*c_attention_s
        support_feats=support_feats+support
        protypes=support_feats.repeat((args.query*args.way,1,1,1))
        ##########################################
        c_attention_q= c_attention.repeat((args.way*args.query, 1, 1))
        c_attention_q= torch.repeat_interleave(c_attention_q, h * w, dim=2).reshape(args.way * args.query * args.way, c, h, w)
        query_sp=self.sp_attention(query_ins,attrib_q,query_ins)
        query_feats=query_sp*c_attention_q
        query_feats=query_feats+query_ins
        relation_pairs = torch.cat((protypes, query_feats), 1).view(-1, c * 2, h, w)
        #print(relation_pairs.shape)
        relations = self.relation_network1(relation_pairs).view(-1, args.way * args.shot)
        #print(relations.shape)
        attrib_s_3=attrib_s_2.repeat((args.way,1,1,1))
        support_feats_1=torch.repeat_interleave(support_feats,args.way,dim=0)
        relations_pairs_fs=torch.cat((support_feats_1,attrib_s_3),dim=1).view(-1, c+self.sem_feat_dim, h, w)
        relations_fs= self.relation_network2(relations_pairs_fs).view(-1, args.way * args.shot)
        return relations,relations_fs








