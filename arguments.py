import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-g','--gamma',type=float,metavar=' ',required=False,default=0.98)

parser.add_argument('-bs','--batch_size',type=int,metavar=' ',required=False,default=128)

parser.add_argument('-bf','--buffer_length',type=int,metavar=' ',required=False,default=131072)

parser.add_argument('-a','--alpha',type=float,metavar=' ',required=False, default=0.7)

parser.add_argument('-b','--beta',type=float,metavar=' ',required=False,default=0.5)

parser.add_argument('-ws','--warm_start',type=int,metavar=' ',required=False,default=100)

parser.add_argument('-seq','--sequence',type=str,metavar=' ',required=True)

parser.add_argument('-ts','--time_steps',type=int,metavar=' ',required=True)

parser.add_argument('-c','--sync_steps',type=int,metavar=' ',required=True)

parser.add_argument('-tf','--train_freq',type=int,metavar=' ',required=True)

parser.add_argument('-m','--num_model',type=int,metavar=' ',required=True)

parser.add_argument('-env','--environment',type=str,metavar=' ',required=True)

args = parser.parse_args()

'python main.py -seq 20merA -bs 64 -ts 1000000 -c 150 -tf 4 -m 3 -env v3.0'

'python main.py -seq 20merA -bs 150 -ts 1000000 -c 120 -tf 4 -m 4 -env v3.1'






