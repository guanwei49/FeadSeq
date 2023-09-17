#!/usr/bin/env python
import os

from pytorch_lightning import seed_everything
import argparse
import os
import time
import warnings

warnings.simplefilter("ignore")



def run(args):
    time_list = []
    print("Creating server and clients ...")
    start = time.time()

    from server import FedSeq
    server = FedSeq(args)


    server.train()

    time_list.append(time.time()-start)

    import numpy as np
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")



if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="BPIC20")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.003,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=True)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.98)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-nc', "--num_clients", type=int, default=5,
                        help="Total number of clients")
    parser.add_argument('-gr', "--global_rounds", type=int, default="60")
    parser.add_argument('-sr', "--sharing_rate", type=float, default="0.3") # 越大，联邦多个client共享的参数越少
    parser.add_argument('-npl', "--num_pre_loss", type=int, default=3,
                        help="Determine if the model converges when obtaining a mask")
    parser.add_argument('-lstdt', "--loss_threshold", type=float, default=0.1,
                        help="Determine if the model converges when obtaining a mask")
    parser.add_argument('-bs', "--batch_size", type=int, default="16")
    parser.add_argument('-seed', "--seed", type=int, default="1000")
    parser.add_argument('-offset', "--offset", type=int, default="0",
                        help="Client id offset")

    parser.add_argument('-ed', "--embed_dim", type=int, default="64",
                        help="embedding dim of the model")
    parser.add_argument('-hd', "--hid_dim", type=int, default="256",
                        help="hidden dim of the model")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    seed_everything(args.seed)
    print("=" * 50)
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Dataset: {}".format(args.dataset))
    print("Using device: {}".format(args.device))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("Global rounds: {}".format(args.global_rounds))
    print("Sharing rate: {}".format(args.sharing_rate))
    print("Number of pre loss: {}".format(args.num_pre_loss))
    print("Loss std threshold: {}".format(args.loss_threshold))
    print("Batch size: {}".format(args.batch_size))
    print("Seed: {}".format(args.seed))
    print("embedding dim: {}".format(args.embed_dim))
    print("hidden dim: {}".format(args.hid_dim))
    print("=" * 50)

    run(args)

