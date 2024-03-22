import os
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from client import client
from GAT_AE import GAT_AE
from utils.data_utils import read_client_data_forSYN


class FeadSeq():
    def __init__(self, args):
        self.args=args
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.embed_dim = args.embed_dim
        self.global_model = None
        self.num_clients = args.num_clients
        self.hid_dim = args.hid_dim
        self.sharing_rate = args.sharing_rate

        self.clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.uploaded_embeds = []

        self.seed=args.seed
        self.offset=args.offset


        self.res_metrics = []


        self.set_clients_and_global_model(client)


        print("Finished creating server and clients.")

        self.Budget = []


    def train(self):
        for i in range(self.global_rounds):
            s_t = time.time()

            print(f"\n-------------Round number: {i}-------------")

            for client in  self.clients:
                if i==0:  #只训练一次mask
                    client.obtain_mask()
                client.set_parameters(self.global_model)  ##将模型参数回传给client

                client.train()

                client.test_metrics()

            self.aggregate_metrics()  #取评估指标的均值
            self.receive_models()  ##收集client模型
            self.aggregate_parameters() ##参数的合并

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])


        # print(self.res_metrics)
        ROOT_DIR = Path(__file__).parent
        res = pd.DataFrame(self.res_metrics)
        res.insert(0, 'global_round', [gr+1 for gr in range(self.global_rounds)])

        print(res)
        print(ROOT_DIR)

        res.to_csv(os.path.join(ROOT_DIR, f'result.csv'), index=False)

    def set_clients_and_global_model(self, clientObj):
        train_unique_attrV = []  # 每个数据集，每个属性的属性值集合
        all_unique_attrV=[]
        client_unique_attrV = []  # 每一个客户中独有的属性值
        all_max_seq_len = []
        all_trace_num = []
        print('Aligning Event Sets')
        for i in range(self.num_clients):
            train_data_forSYN,test_data_forSYN = read_client_data_forSYN(self.dataset, i+ self.offset, 0.10)
            all_unique_attrV.append(test_data_forSYN.unique_attrV)
            train_unique_attrV.append(train_data_forSYN.unique_attrV)
            all_max_seq_len.append(max(train_data_forSYN.max_seq_len,test_data_forSYN.max_seq_len))
            all_trace_num.append(len(train_data_forSYN))
            for k in all_unique_attrV[i].keys():
                all_unique_attrV[i][k]= all_unique_attrV[i][k].union(train_unique_attrV[i][k])

        common_attrV = {}  # 每个属性集合中，数据集共同的属性值
        for k in train_unique_attrV[0].keys():
            for i in range(self.num_clients):
                if k not in common_attrV.keys():
                    common_attrV[k] = train_unique_attrV[i][k]
                else:
                    common_attrV[k] = common_attrV[k].intersection(train_unique_attrV[i][k])

        self.global_model = {}
        self.common_attrV = common_attrV

        for i in range(self.num_clients):
            temp = {}
            for k in all_unique_attrV[0].keys():
                temp[k] = sorted(list(all_unique_attrV[i][k] - common_attrV[k]))
            client_unique_attrV.append(temp)

        for k in self.common_attrV.keys():
            self.common_attrV[k] = sorted(list(self.common_attrV[k]))
        self.common_attribute_dims = [len(v) for v in self.common_attrV.values()]
        self.global_model['GAT_AEs'] = GAT_AE(self.embed_dim, self.common_attribute_dims, max(all_max_seq_len),
                                              hidden_dim=self.hid_dim).to(self.device)

        common_embeds = []
        for i, dim in enumerate(self.common_attribute_dims):
            common_embeds.append(nn.Embedding(dim + 1, self.embed_dim))  # 给占位符 0 的 位置 所以要加1
        self.global_model['shared_embeds'] = nn.ModuleList(common_embeds).to(self.device)

        for i in range(self.num_clients):
            client = clientObj(self.args,
                               id=i+self.offset,
                               train_samples=all_trace_num[i],  ##第i个数据集中轨迹个数
                               max_seq_len=all_max_seq_len[i],  ##第i个数据集中最长轨迹长度
                               common_attrV=common_attrV,  ##数据集共同的属性值, 字典类型
                               unique_attrV=client_unique_attrV[i],  ##第i个数据集中属性以及其独有的属性值， 字典类型
                               GAT_AEs=self.global_model['GAT_AEs'],
                               shared_embeds=self.global_model['shared_embeds']
                               )
            self.clients.append(client)

    def aggregate_metrics(self):
        metrics = []
        for client in self.clients:
            metrics.append(client.metrics)

        metrics = pd.concat(metrics)
        self.res_metrics.append(metrics.mean())
        print(metrics.mean())

    def receive_models(self):
        active_clients = self.clients

        self.uploaded_ids = []
        self.uploaded_counts = []
        self.uploaded_models = []
        self.uploaded_embeds = []
        model_params_masks = []
        common_embeds_params_masks = []
        tot_samples = 0
        upload_model_count_mask = []
        upload_embed_count_mask = []
        self.model_params_weight_masks = []
        self.common_embeds_params_weight_masks = []
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_counts.append(client.train_samples)
            self.uploaded_models.append(client.model)
            self.uploaded_embeds.append(client.shared_embeds)
            model_params_masks.append(client.model_params_mask)
            common_embeds_params_masks.append(client.shared_embeds_params_mask)
        for mask_index in range(len(model_params_masks[0])):
            mask_list = []
            for i in range(len(model_params_masks)):
                mask_list.append(model_params_masks[i][mask_index] * self.uploaded_counts[i])
            upload_model_count_mask.append(torch.sum(torch.stack(mask_list, 0), 0))
        for mask_index in range(len(common_embeds_params_masks[0])):
            mask_list = []
            for i in range(len(common_embeds_params_masks)):
                mask_list.append(common_embeds_params_masks[i][mask_index] * self.uploaded_counts[i])
            upload_embed_count_mask.append(torch.sum(torch.stack(mask_list, 0), 0))

        for i in range(len(model_params_masks)):
            weight_masks = []
            for mask_index in range(len(model_params_masks[0])):
                model_mask = model_params_masks[i][mask_index] * self.uploaded_counts[i] / upload_model_count_mask[
                    mask_index]
                model_mask[model_mask.isnan()] = 0  # 避免分母为0造成的nan
                weight_masks.append(model_mask)
            self.model_params_weight_masks.append(weight_masks)

        for i in range(len(common_embeds_params_masks)):
            weight_masks = []
            for mask_index in range(len(common_embeds_params_masks[0])):
                model_mask = common_embeds_params_masks[i][mask_index] * self.uploaded_counts[i] / \
                             upload_embed_count_mask[
                                 mask_index]
                model_mask[model_mask.isnan()] = 0  # 避免分母为0造成的nan
                weight_masks.append(model_mask)
            self.common_embeds_params_weight_masks.append(weight_masks)

        # for i, w in enumerate(self.uploaded_weights):
        #     self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        for param in self.global_model['GAT_AEs'].parameters():  ##模型的所有参数设置为0
            param.data.zero_()
        for param in self.global_model['shared_embeds'].parameters():  ##模型的所有参数设置为0
            param.data.zero_()

        for w, client_model in zip(self.model_params_weight_masks, self.uploaded_models):
            self.add_parameters(self.global_model['GAT_AEs'], w, client_model)
        for w, client_model in zip(self.common_embeds_params_weight_masks, self.uploaded_embeds):
            self.add_parameters(self.global_model['shared_embeds'], w, client_model)

    def add_parameters(self, server_model, w, client_model):
        for server_param, client_param, weight_mask in zip(server_model.parameters(), client_model.parameters(), w):
            server_param.data += client_param.data.clone() * weight_mask
