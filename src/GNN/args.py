import argparse

# Initialize the parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# General Arguments
parser.add_argument('-id', '--device_id', default='1', type=str,
                    help='Set the device (GPU ids)')
parser.add_argument('-da', '--dataset', default='/dataset/B-dataset', type=str,
                    help='Set the data set for training')

parser.add_argument('-ll', '--llm_rep_path', default='llm_rep_final_dataset.pkl', type=str,
                    help='LLM generated representations')

parser.add_argument('-di', '--Disease_mapper', default='Disease_B_mapper.pkl', type=str,
                    help=' Map disease number to its name ')

parser.add_argument('-dr', '--Drug_mapper', default='Drug_B_mapper.pkl', type=str,
                    help='Map drug number to its name')

parser.add_argument('-sp', '--saved_path', default='B_normalize_seed2', type=str,
                    help='Path to save training results')

parser.add_argument('-se', '--seed', default=0, type=int,
                    help='Global random seed to be used')

# Training Arguments
parser.add_argument('-pr', '--print_every', default=1, type=int,
                    help='The number of epochs to print a training record')
parser.add_argument('-fo', '--nfold', default=5, type=int,
                    help='The number of k in k-fold cross validation')
parser.add_argument('-ep', '--epoch', default=200, type=int,
                    help='The number of epochs for model training')
parser.add_argument('-bs', '--batch_size', default=2048, type=int,
                    help='The size of a batch to be used')
parser.add_argument('-lr', '--learning_rate', default=0.005, type=float,
                    help='Learning rate to be used in optimizer')
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float,
                    help='weight decay to be used')
parser.add_argument('-ck', '--check_metric', default='auc', type=str, choices=['loss', 'auc', 'aupr'],
                    help='Metric to check')

# Model Arguments
parser.add_argument('-k', '--k', default=15, type=int,
                    help='The number of topk similarities to be binarized')
parser.add_argument('-ag', '--aggregate_type', default='BiTrans', type=str, choices=['sum', 'mean', 'Linear', 'BiTrans'],
                    help='The type of aggregator to be used for aggregating meta-path instances')
parser.add_argument('-tk', '--topk', default=1, type=int,
                    help='The topk instance predictions to be chosen')
parser.add_argument('-hf', '--hidden_feats', default=64, type=int,
                    help='The dimension of hidden tensor in the model')
parser.add_argument('-nl', '--num_layer', default=2, type=int,
                    help='The number of graph embedding layers to be used')
parser.add_argument('-dp', '--dropout', default=0.2, type=float,
                    help='The rate of dropout layer')
parser.add_argument('-bn', '--batch_norm', action='store_true',
                    help='Use batch normalization')
parser.add_argument('-sk', '--skip', default=False, type=bool,
                    help='Skip option')
parser.add_argument('-mil', '--mil', default=False, type=bool,
                    help='MIL option')
parser.add_argument('-ip', '--ins_predict', default=False, type=bool,
                    help='Instance prediction option')

# Parse arguments
args = parser.parse_args()

# Post-process saved path
args.saved_path = args.saved_path + '_' + str(args.seed)
