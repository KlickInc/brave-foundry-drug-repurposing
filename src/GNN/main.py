import argparse
import pandas as pd 
import numpy as np 
import os
import torch
import pandas as pd
import numpy as np
from warnings import simplefilter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset
from load_data import load_data, load_graph, remove_graph, \
    get_data_loaders
from model_wollm import Model
from utils import get_metrics, get_metrics_auc, set_seed, \
    plot_result_auc, plot_result_aupr, checkpoint
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pickle
from args import args

with open(args.llm_rep_path, 'rb') as handle:
        dict_p = pickle.load(handle)

with open(args.Disease_mapper, 'rb') as handle:
       mapper_disease = pickle.load(handle)
  
with open(args.Drug_mapper, 'rb') as handle:
       mapper_drug= pickle.load(handle)
       
# Load the mappings of drug and disease IDs to their names
dr = pd.read_csv(f'{args.dataset}/drug.csv')
di = pd.read_csv(f'{args.dataset}/disease.csv')

def compute_metrics(labels, values):
    # Predicted classes based on threshold of 0.5
    predicted = [1 if v > 0.5 else 0 for v in values]

    # Calculate metrics
    auc = roc_auc_score(labels, values)
    aupr = average_precision_score(labels, values)
    acc = accuracy_score(labels, predicted)
    f1 = f1_score(labels, predicted)
    pre = precision_score(labels, predicted)
    rec = recall_score(labels, predicted)

    # Specificity: TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(labels, predicted).ravel()
    spec = tn / (tn + fp)

    return auc,aupr, acc,  f1, pre,  rec, spec


def val(args, model, val_loader, val_label,
        g, feature, device):
    model.eval()
    pred_val = torch.zeros(val_label.shape).to(device)
    with torch.no_grad():
        for i, data_ in enumerate(val_loader):
            x_val, y_val = data_[0].to(device), data_[1].to(device)
            drug_list = x_val[:,0,0] # drug 
            disease_list  = x_val[:,0,2] # disease 
            combined_tensor_list = [
                    dict_p[f"{int(a)}_{mapper_drug[int(a)]}_{int(b)}_{mapper_disease[int(b)]}"]
                    for a, b in zip(drug_list, disease_list)
                ]
            #combined_tensor_list = np.array([dict_p[f'{a}_{b}'] for a, b in zip(drug_list.cpu().numpy(), disease_list.cpu().numpy())])
            combined_tensor_list=np.array(combined_tensor_list)
            combined_tensor_list = combined_tensor_list.reshape(combined_tensor_list.shape[0],combined_tensor_list.shape[1])
            combined_tensor_list = torch.tensor(combined_tensor_list).to(device)
            
            pred_, attn_ = model(g, feature, x_val,combined_tensor_list)
            pred_ = pred_.squeeze(dim=1)
            score_ = torch.sigmoid(pred_)
            pred_val[args.batch_size * i: args.batch_size * i + len(y_val)] = score_.detach()
    AUC_val, AUPR_val = get_metrics_auc(val_label.cpu().detach().numpy(), pred_val.cpu().detach().numpy())
    return AUC_val, AUPR_val, pred_val


def train():
    simplefilter(action='ignore', category=UserWarning)
    print('Arguments: {}'.format(args))
    set_seed(args.seed)

    if not os.path.exists(f'result/{args.dataset}'):
        os.makedirs(f'result/{args.dataset}')
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    argsDict = args.__dict__
    with open(os.path.join(args.saved_path, 'setting.txt'), 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    if args.device_id != 'cpu':
        print('Training on GPU')
        device = torch.device('cuda:{}'.format(args.device_id))
    else:
        print('Training on CPU')
        device = torch.device('cpu')

    g, data, label = load_data(args)

    data = torch.tensor(data).to(device)
    label = torch.tensor(label).float().to(device)
    
    kf = StratifiedKFold(args.nfold, shuffle=True, random_state=args.seed)
    fold = 1

    pred_result = np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))

    for (train_idx, val_idx) in kf.split(data.cpu().numpy(), label.cpu().numpy()):
        print('{}-Fold Cross Validation: Fold {}'.format(args.nfold, fold))
        train_data = data[train_idx]
        train_label = label[train_idx]
        
        val_data = data[val_idx]
        val_label = label[val_idx]
        val_drug_id = [datapoint[0][0].item() for datapoint in val_data]
        val_disease_id = [datapoint[0][-1].item() for datapoint in val_data]
        dda_idx = torch.where(val_label == 1)[0].cpu().numpy()
        val_dda_drugid = np.array(val_drug_id)[dda_idx]
        val_dda_disid = np.array(val_disease_id)[dda_idx]

        g_train = g
        g_train = remove_graph(g_train, val_dda_drugid.tolist(), val_dda_disid.tolist()).to(device)

        feature = {'drug': g_train.nodes['drug'].data['h'],
                   'disease': g_train.nodes['disease'].data['h']}
        train_loader = get_data_loaders(TensorDataset(train_data, train_label), args.batch_size,
                                        shuffle=True, drop=True)

        val_loader = get_data_loaders(TensorDataset(val_data, val_label), args.batch_size, shuffle=False)

        model = Model(g.etypes,
                      {'drug': feature['drug'].shape[1], 'disease': feature['disease'].shape[1]},
                      hidden_feats=args.hidden_feats,
                      num_emb_layers=args.num_layer,
                      agg_type=args.aggregate_type,
                      dropout=args.dropout,
                      bn=args.batch_norm,
                      k=args.topk )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(torch.where(train_label == 0)[0]) /
                                                                       len(torch.where(train_label == 1)[0])))
        print('BCE loss pos weight: {:.3f}'.format(
            len(torch.where(train_label == 0)[0]) / len(torch.where(train_label == 1)[0])))

        record_list = []
        print_list = []

        for epoch in range(1, args.epoch): # chnage 
   
            total_loss = 0


            pred_train, label_train = torch.zeros(train_label.shape).to(device), \
                                      torch.zeros(train_label.shape).to(device)
            for i, data_ in enumerate(train_loader):
                model.train()
                x_train, y_train = data_[0].to(device), data_[1].to(device)
                drug_list = x_train[:,0,0] # drug 
                disease_list  = x_train[:,0,2] # disease 
       

                llm_rep = [
                    dict_p[f"{int(a)}_{mapper_drug[int(a)]}_{int(b)}_{mapper_disease[int(b)]}"]
                    for a, b in zip(drug_list, disease_list)
                ]

                llm_rep=np.array(llm_rep)
                llm_rep = llm_rep.reshape(llm_rep.shape[0],llm_rep.shape[1])
                llm_rep = torch.tensor(llm_rep).to(device)


                pred1, attn = model(g_train, feature, x_train,llm_rep)
                pred1 = pred1.squeeze(dim=1)
                score = torch.sigmoid(pred1)
                optimizer.zero_grad()
   
                    
                all_loss = criterion(pred1, y_train)

                all_loss.backward()
                optimizer.step()
                total_loss += all_loss.item() / len(train_loader)

                pred_train[args.batch_size * i: args.batch_size * i + len(y_train)] = score.detach()
                label_train[args.batch_size * i: args.batch_size * i + len(y_train)] = y_train.detach()

            AUC_train, AUPR_train = get_metrics_auc(label_train.cpu().detach().numpy(),
                                                    pred_train.cpu().detach().numpy())
     
            AUC_val, AUPR_val, pred_val = val(args, model, val_loader, val_label, g_train, feature, device)
            if epoch % args.print_every == 0:
                print('Epoch train LOVENet {} Loss: {:.5f}; Train: AUC {:.3f}, AUPR {:.3f};'
                     ' Val: AUC {:.3f}, AUPR {:.3f}'.format(epoch, total_loss, AUC_train,
                                                             AUPR_train, AUC_val, AUPR_val))
                
        
                AUC_val, AUPR_val, Acc, F1, Pre, Rec, Spec = compute_metrics(val_label.cpu().detach().numpy(), pred_val.cpu().detach().numpy())
                record_list.append([total_loss, AUC_train, AUPR_train, AUC_val, AUPR_val, Acc, F1, Pre, Rec, Spec])
                
            print_list.append([total_loss, AUC_train, AUPR_train])
            m = checkpoint(args, model, print_list, [total_loss, AUC_train, AUPR_train], fold)
            if m:
                best_model = m


        AUC_val, AUPR_val, pred_val = val(args, best_model, val_loader, val_label, g_train, feature, device)
        pred_result[val_drug_id, val_disease_id] = pred_val.cpu().detach().numpy()
        pd.DataFrame(np.array(record_list),
                     columns=['Loss', 'AUC_train', 'AUPR_train',
                              'AUC_val', 'AUPR_val','ACC', 'F1', 'Pre', 'Rec', 'Spec']).to_csv(os.path.join(args.saved_path,
                                                                          'training_score_{}.csv'.format(fold)),
                                                             index=False)
        fold += 1
        # break


    val_label_array = label.cpu().detach().numpy()
    pred_result_array = pred_result.flatten()

    final_result = pd.DataFrame({'Actual': val_label_array, 'Prediction': pred_result_array})
    final_result.to_csv(os.path.join(args.saved_path, 'Final_result.csv'))



if __name__ == '__main__':
    train()
        