import torch

from layer import *
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pickle

    
class Model(nn.Module):
    def __init__(self, etypes, in_feats, hidden_feats,
                 num_emb_layers, agg_type='sum', k=0, dropout=0., bn=False,
                 skip=True, mil=True):
        """
        Parameters
        ----------
        etypes : list
            e.g: ['disease-disease', 'drug-disease', 'drug-drug']
        in_feats : dict[str, int]
            Input feature size for each node type.
        hidden_feats : int
            Hidden feature size.
        num_emb_layers : int
            Number of embedding layers to be used.
        agg_type : string
            Type of meta-path aggregator to be used, including "sum", "average", "linear", and "RotatE".
        dropout : float
            The dropout rate to be used.
        bn : bool
            Whether to use batch normalization layer.
        """
        super(Model, self).__init__()
        self.lin_transform = HeteroLinear(in_feats, hidden_feats, dropout, bn)
        self.graph_embedding = nn.ModuleDict()
        for l in range(num_emb_layers):
            self.graph_embedding['Layer_{}'.format(l)] = Node_Embedding(etypes,
                                                                        hidden_feats,
                                                                        hidden_feats,
                                                                        dropout, bn)
        self.layer_attention_drug = LayerAttention(hidden_feats,
                                                   hidden_feats)
        self.layer_attention_dis = LayerAttention(hidden_feats,
                                                  hidden_feats)
        self.aggregator = MetaPathAggregator(hidden_feats, hidden_feats, agg_type)
        self.mil_layer = MILNet(hidden_feats, hidden_feats)
        self.bag_predict = MLP(hidden_feats * 2 , dropout=0.)


        self.llm_head_layer = nn.Linear(32000, hidden_feats).to('cuda:1')
        


        if k > 0 and agg_type == 'BiTrans':
            self.ins_predict = InstanceNet(hidden_feats, k)
        else:
            self.ins_predict = None
        self.skip = skip
        self.mil = mil
        

    def forward(self, g, feature, mp_ins,llm_rep):
        """
        Parameters
        ----------
        g : dgl.graph
            Heterogeneous graph representing the drug-disease network.
        feature : dict[node_types, feature_tensors]
            Initialized node features of g.
        mp_ins : torch.tensor
            Bags of meta-path instances.
            
        llm_rep: torch.tensor
           LLM generated representations
        """
        h_integrated_drug, h_integrated_dis = [], []

        h = self.lin_transform(feature)
        h_integrated_drug.append(h['drug'])
        h_integrated_dis.append(h['disease'])
        for emb_layer in self.graph_embedding:
            h = self.graph_embedding[emb_layer](g, h)
            h_integrated_drug.append(h['drug'])
            h_integrated_dis.append(h['disease'])
        if self.skip:
            h = dict(zip(['drug', 'disease'],
                         [torch.stack(h_integrated_drug, dim=1),
                          torch.stack(h_integrated_dis, dim=1)]))
            h['drug'] = self.layer_attention_drug(h['drug'])
            h['disease'] = self.layer_attention_dis(h['disease'])

        ins_emb = self.aggregator(h, mp_ins)
        if self.mil:
            bag_emb, attn = self.mil_layer(ins_emb)
            llm_rep = llm_rep.to(bag_emb.dtype)

            llm_rep = self.llm_head_layer(llm_rep)
            
            projected_llm_normalized = torch.nn.functional.normalize(llm_rep, p=2, dim=1)
            kg_embeddings_normalized = torch.nn.functional.normalize(bag_emb, p=2, dim=1)
           
            result = torch.cat((kg_embeddings_normalized,projected_llm_normalized),dim=1)


        pred_bag = self.bag_predict(result)

        return   pred_bag , attn
