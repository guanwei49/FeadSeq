import copy
import itertools
import random

import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score
from torch import nn
import numpy as np
import time

from torch_geometric.data import Batch, Data
from tqdm import tqdm

from GAT_AE import GAT_AE
from utils.data_utils import read_client_data


class client():
    def __init__(self, args, id, train_samples, **kwargs):
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.train_samples = train_samples
        #训练相关
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        self.common_attrV = kwargs['common_attrV']  # dict ：{属性：属性值集合} 不同客户之间 共同的属性值
        self.unique_attrV = kwargs['unique_attrV']  # dict ：{属性：属性值集合} 自己的独有属性值
        self.max_seq_len = kwargs['max_seq_len']  # 最长轨迹长度
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        #mask相关
        self.sharing_rate= args.sharing_rate
        self.num_pre_loss =args.num_pre_loss
        self.loss_threshold =args.loss_threshold

        self.learning_rate_decay = args.learning_rate_decay

        #模型相关
        self.embed_dim = args.embed_dim
        self.hid_dim=args.hid_dim

        self.loss = nn.CrossEntropyLoss()
        self.common_attribute_dims=[len(v) for v in self.common_attrV.values()]
        self.model= copy.deepcopy(kwargs['GAT_AEs'])

        self.model_params_mask = None  # 保存模型的mask
        self.shared_embeds_params_mask = None # 保存模型的mask

        unique_embed_list=[]

        self.unique_attribute_dims = [len(v) for v in self.unique_attrV.values()]

        for i, dim in enumerate(self.unique_attribute_dims):
            unique_embed_list.append(nn.Embedding(dim , self.embed_dim))
        self.shared_embeds =  copy.deepcopy(kwargs['shared_embeds'])   #从gloabl复制过来
        self.standalone_embeds = nn.ModuleList(unique_embed_list).to(self.device)

        unique_linear_list=[]
        for i, dim in enumerate(self.unique_attribute_dims):
            if i ==0:
                unique_linear_list.append(nn.Linear( 2*self.hid_dim + self.embed_dim, dim))
            else:
                unique_linear_list.append(nn.Linear( 2*self.hid_dim + 2*self.embed_dim , dim ))
        self.standalone_linears = nn.ModuleList(unique_linear_list).to(self.device)

        self.optimizer = torch.optim.SGD(itertools.chain(self.model.parameters(), self.shared_embeds.parameters()
                                                         , self.standalone_embeds.parameters(), self.standalone_linears.parameters()), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )


    def obtain_mask(self):
        '''
        取梯度最小的一些值的mask设为True，梯度大的值mask设为False（parameter设为0）  梯度大的不参与联邦
        :return:
        '''
        traindataset = self.load_train_data()
        self.model.train()
        self.standalone_linears.train()
        self.shared_embeds.train()
        self.standalone_embeds.train()

        self.model_params_mask = []  # 保存模型的mask
        self.shared_embeds_params_mask = []  # 保存模型的mask

        init_model_params = []  # server端初始模型参数
        init_common_embeds_params = []  # server端初始模型参数
        init_unique_embeds_params = []
        init_unique_linears_params = []

        for param in self.model.parameters():
            init_model_params.append(param.data.clone())
        for param in self.shared_embeds.parameters():
            init_common_embeds_params.append(param.data.clone())
        # 私有模型
        for param in self.standalone_linears.parameters():
            init_unique_linears_params.append(param.data.clone())
        for param in self.standalone_embeds.parameters():
            init_unique_embeds_params.append(param.data.clone())

        Xs = []
        for i, dim in enumerate(
                traindataset.attribute_dims):  # self.traindataset.attribute_dims 和 self.all_attrV的各项长度一致
            Xs.append(torch.LongTensor(traindataset.features[i]))

        print("*" * 10 + "training——mask" + "*" * 10 + 'id:' + str(self.id))
        indexes = [i for i in range(len(traindataset))]
        epoch=0
        losses=[]
        while True:
            print('mask epoch:{}'.format(epoch))
            random.shuffle(indexes)
            train_loss = 0.0
            train_num = 0
            for bathc_i in tqdm(range(self.batch_size, len(indexes) + 1, self.batch_size)):
                this_batch_indexes = indexes[bathc_i - self.batch_size:bathc_i]
                nodes_list = [traindataset.node_xs[i] for i in this_batch_indexes]
                edge_indexs_list = [traindataset.edge_indexs[i] for i in this_batch_indexes]
                Xs_list = []
                graph_batch_list = []
                for i in range(len(traindataset.attribute_dims)):
                    Xs_list.append(Xs[i][this_batch_indexes].to(self.device))
                    graph_batch = Batch.from_data_list([Data(x=nodes_list[b][i], edge_index=edge_indexs_list[b])
                                                        for b in range(len(nodes_list))])
                    graph_batch_list.append(graph_batch.to(self.device))
                mask = torch.tensor(traindataset.mask[this_batch_indexes]).to(
                    self.device)  # [batch_size,max_seq_len]

                for k, graph in enumerate(graph_batch_list):  # 对图的顶点做 embedding 并放入设备
                    graph_batch_list[k] = graph.to(self.device)
                    temp_mask = graph.x > self.common_attribute_dims[k]
                    temp = torch.Tensor(graph.x.__len__(), self.embed_dim).to(self.device)
                    temp[temp_mask] = self.standalone_embeds[k](graph.x[temp_mask] - self.common_attribute_dims[k] - 1)
                    temp[~temp_mask] = self.shared_embeds[k](graph.x[~temp_mask])
                    graph.x = temp

                # attr_reconstruction_outputs  list : 其中元素形状 [batch_size,max_seq_len,self.common_attribute_dims[i]]
                attr_reconstruction_outputs, middles = self.model(graph_batch_list, mask, len(this_batch_indexes))

                for i, middle in enumerate(middles):
                    unique_reconstruction_out = self.standalone_linears[i](middle)
                    unique_reconstruction_out = torch.cat(
                        (torch.zeros((self.batch_size, 1, self.unique_attribute_dims[i])).to(
                            self.device), unique_reconstruction_out), 1)
                    attr_reconstruction_outputs[i] = torch.cat(
                        (attr_reconstruction_outputs[i], unique_reconstruction_out), 2)

                self.optimizer.zero_grad()

                loss = 0.0
                mask[:, 0] = False  # 除了每一个属性的起始字符之外,其他重建误差
                for i in range(len(traindataset.attribute_dims)):
                    # --------------
                    # 除了每一个属性的起始字符之外,其他重建误差
                    # ---------------
                    pred = attr_reconstruction_outputs[i][mask]
                    true = Xs_list[i][mask]
                    loss += self.loss(pred, true)
                train_loss += loss.item()
                train_num += 1
                loss.backward()
                self.optimizer.step()
            self.learning_rate_scheduler.step()
            ## 计算一个epoch在训练集上的损失和精度
            train_loss_epoch = train_loss / train_num
            losses.append(train_loss_epoch)
            print(f"[loss: {train_loss_epoch:3f}]")
            # train the weight until convergence
            epoch += 1
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.loss_threshold:
                print('Client:', self.id, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                      '\tmask total epochs:', epoch)
                break


        for i, (param, init_param) in enumerate(zip(self.model.parameters(), init_model_params)):  # 计算mask
            param_dif = torch.abs(param.data.clone() - init_param.data.clone())
            param_flat = param_dif.flatten()
            value = param_flat.sort().values[
                min(int(len(param_flat) * self.sharing_rate), len(param_flat) - 1)]
            model_param_mask = torch.zeros_like(param_dif, device=self.device, dtype=torch.bool)
            model_param_mask[param_dif <= value] = True  # 变化小的 设置mask为 True ，True时候，此位置的参数保留
            self.model_params_mask.append(model_param_mask)


        for i, (param, init_param) in enumerate(
                zip(self.shared_embeds.parameters(), init_common_embeds_params)):  # 计算mask
            param_dif = torch.abs(param.data.clone() - init_param.data.clone())
            value = param_dif.sort().values[ :, min(int(len(param_dif[0]) * self.sharing_rate), len(param_dif[0]) - 1)]
            model_param_mask = torch.zeros_like(param_dif, device=self.device, dtype=torch.bool)
            value = value.unsqueeze(1).repeat((1, len(param_dif[0])))
            model_param_mask[param_dif <= value] = True  # 变化小的 设置mask为 True ，True时候，此位置的参数保留
            self.shared_embeds_params_mask.append(model_param_mask)

        print('sharing:{}'.format(
            self.model_params_mask[0].sum() / np.product(np.array(self.model_params_mask[0].shape))))
        print('sharing:{}'.format(self.shared_embeds_params_mask[0].sum() / np.product(np.array(self.shared_embeds_params_mask[0].shape))))

        for new_param, old_param in zip(init_model_params, self.model.parameters()):  # 重新赋值成初始参数
            old_param.data = new_param.data.clone()
        for new_param, old_param in zip(init_common_embeds_params, self.shared_embeds.parameters()):
            old_param.data = new_param.data.clone()
        for new_param, old_param in zip(init_unique_linears_params, self.standalone_linears.parameters()):
            old_param.data = new_param.data.clone()
        for new_param, old_param in zip(init_unique_embeds_params, self.standalone_embeds.parameters()):
            old_param.data = new_param.data.clone()
        self.optimizer.param_groups[0]['lr'] = self.learning_rate  # 重置learning rate

    def load_train_data(self):
        train_data = read_client_data(self.dataset,self.id ,0.00,self.common_attrV,self.unique_attrV)
        return train_data

    def load_test_data(self):
        train_data = read_client_data(self.dataset,self.id, 0.10,self.common_attrV,self.unique_attrV)
        return train_data

    def set_parameters(self, model):
        for new_param, old_param,model_mask in zip(model['GAT_AEs'].parameters(), self.model.parameters(),self.model_params_mask):
            old_param.data = new_param.data.clone()*model_mask  +  old_param.data.clone() * ~model_mask
        for new_param, old_param,model_mask in zip(model['shared_embeds'].parameters(), self.shared_embeds.parameters(), self.shared_embeds_params_mask):
            old_param.data = new_param.data.clone()*model_mask +  old_param.data.clone() * ~model_mask

    def train(self):
        traindataset = self.load_train_data()
        self.model.train()
        self.standalone_linears.train()
        self.shared_embeds.train()
        self.standalone_embeds.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs

        Xs = []
        for i, dim in enumerate(traindataset.attribute_dims): # self.traindataset.attribute_dims 和 self.all_attrV的各项长度一致
            Xs.append(torch.LongTensor(traindataset.features[i]))

        print("*" * 10 + "training" + "*" * 10+'id:'+str(self.id)+"*" * 10+'LR:'+str(self.optimizer.param_groups[0]['lr']))
        indexes = [i for i in range(len(traindataset))]
        # indexes = indexes[:100]
        random.shuffle(indexes)  # 打乱顺序

        for epoch in range(max_local_epochs):
            train_loss = 0.0
            train_num = 0

            for bathc_i in tqdm(range(self.batch_size, len(indexes) + 1, self.batch_size)):
                this_batch_indexes = indexes[bathc_i - self.batch_size:bathc_i]
                nodes_list = [traindataset.node_xs[i] for i in this_batch_indexes]
                edge_indexs_list = [traindataset.edge_indexs[i] for i in this_batch_indexes]
                Xs_list = []
                graph_batch_list = []
                for i in range(len(traindataset.attribute_dims)):
                    Xs_list.append(Xs[i][this_batch_indexes].to(self.device))
                    graph_batch = Batch.from_data_list([Data(x=nodes_list[b][i], edge_index=edge_indexs_list[b])
                                                        for b in range(len(nodes_list))])
                    graph_batch_list.append(graph_batch.to(self.device))
                dataset_mask = torch.tensor(traindataset.mask[this_batch_indexes]).to(self.device)  # [batch_size,max_seq_len]

                for k, graph in enumerate(graph_batch_list): #对图的顶点做 embedding 并放入设备
                    graph_batch_list[k] = graph.to(self.device)
                    temp_mask = graph.x> self.common_attribute_dims[k]
                    temp = torch.Tensor(graph.x.__len__(), self.embed_dim).to(self.device)
                    temp[temp_mask]=self.standalone_embeds[k](graph.x[temp_mask] - self.common_attribute_dims[k] - 1)
                    temp[~temp_mask]=self.shared_embeds[k](graph.x[~temp_mask])
                    graph.x = temp

                #attr_reconstruction_outputs  list : 其中元素形状 [batch_size,max_seq_len,self.common_attribute_dims[i]]
                attr_reconstruction_outputs,middles = self.model(graph_batch_list,dataset_mask,len(this_batch_indexes))


                for i, middle in enumerate(middles):
                    unique_reconstruction_out = self.standalone_linears[i](middle)
                    unique_reconstruction_out = torch.cat((torch.zeros((self.batch_size, 1, self.unique_attribute_dims[i])).to(
                            self.device),unique_reconstruction_out),1)
                    attr_reconstruction_outputs[i]= torch.cat((attr_reconstruction_outputs[i],unique_reconstruction_out),2)

                self.optimizer.zero_grad()

                loss = 0.0
                dataset_mask[:, 0] = False  # 除了每一个属性的起始字符之外,其他重建误差
                for i in range(len(traindataset.attribute_dims)):
                    # --------------
                    # 除了每一个属性的起始字符之外,其他重建误差
                    # ---------------
                    pred = attr_reconstruction_outputs[i][dataset_mask]
                    true = Xs_list[i][dataset_mask]
                    loss += self.loss(pred, true)

                train_loss += loss.item()
                train_num += 1
                loss.backward()

                self.optimizer.step()
            ## 计算一个epoch在训练集上的损失和精度
            train_loss_epoch = train_loss / train_num
            print(f"[Epoch {epoch + 1:{len(str(max_local_epochs))}}/{max_local_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]")

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self):
        dataset = self.load_test_data()
        self.model.eval()
        self.standalone_linears.eval()
        self.shared_embeds.eval()
        self.standalone_embeds.eval()

        with torch.no_grad():
            final_res = []
            attribute_dims = dataset.attribute_dims

            Xs = []
            for i, dim in enumerate(dataset.attribute_dims):
                Xs.append(torch.LongTensor(dataset.features[i]))

            print("*" * 10 + "detecting" + "*" * 10+'id:'+str(self.id))

            pre = 0
            for bathc_i in tqdm(range(self.batch_size, len(dataset) + self.batch_size, self.batch_size)):
                if bathc_i <= len(dataset):
                    this_batch_indexes = list(range(pre, bathc_i))
                else:
                    this_batch_indexes = list(range(pre, len(dataset)))

                nodes_list = [dataset.node_xs[i] for i in this_batch_indexes]
                edge_indexs_list = [dataset.edge_indexs[i] for i in this_batch_indexes]
                Xs_list = []
                graph_batch_list = []

                for i in range(len(dataset.attribute_dims)):
                    Xs_list.append(Xs[i][this_batch_indexes].to(self.device))
                    graph_batch = Batch.from_data_list([Data(x=nodes_list[b][i], edge_index=edge_indexs_list[b])
                                                        for b in range(len(nodes_list))])
                    graph_batch_list.append(graph_batch.to(self.device))
                dataset_mask = torch.tensor(dataset.mask[this_batch_indexes]).to(
                    self.device)  # [batch_size,max_seq_len]

                for k, graph in enumerate(graph_batch_list):  # 对图的顶点做 embedding 并放入设备
                    graph_batch_list[k] = graph.to(self.device)
                    temp_mask = graph.x > self.common_attribute_dims[k]
                    temp = torch.Tensor(graph.x.__len__(), self.embed_dim).to(self.device)
                    temp[temp_mask] = self.standalone_embeds[k](graph.x[temp_mask] - self.common_attribute_dims[k] - 1)
                    temp[~temp_mask] = self.shared_embeds[k](graph.x[~temp_mask])
                    graph.x = temp

                # attr_reconstruction_outputs  list : 其中元素形状 [batch_size,max_seq_len,self.common_attribute_dims[i]]
                attr_reconstruction_outputs, middles = self.model(graph_batch_list, dataset_mask,
                                                                  len(this_batch_indexes))

                for i, middle in enumerate(middles):
                    unique_reconstruction_out = self.standalone_linears[i](middle)
                    unique_reconstruction_out = torch.cat(
                        (torch.zeros((len(this_batch_indexes), 1, self.unique_attribute_dims[i])).to(
                            self.device), unique_reconstruction_out), 1)
                    attr_reconstruction_outputs[i] = torch.cat(
                        (attr_reconstruction_outputs[i], unique_reconstruction_out), 2)

                for attr_index in range(len(attribute_dims)):
                    attr_reconstruction_outputs[attr_index] = torch.softmax(attr_reconstruction_outputs[attr_index],
                                                                            dim=2)

                this_res = []
                for attr_index in range(len(attribute_dims)):
                    # 取比实际出现的属性值大的其他属性值的概率之和
                    temp = attr_reconstruction_outputs[attr_index]
                    index = Xs_list[attr_index].unsqueeze(2)
                    probs = temp.gather(2, index)
                    temp[(temp <= probs)] = 0
                    res = temp.sum(2)
                    res = res * (dataset_mask)
                    this_res.append(res)

                final_res.append(torch.stack(this_res, 2))

                pre = bathc_i


            attr_level_abnormal_scores = np.array(torch.cat(final_res, 0).detach().cpu())
            trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
            event_level_abnormal_scores = attr_level_abnormal_scores.max((2))

            def cal_best_PRF(y_true, probas_pred):
                '''
                计算在任何阈值下，最好的precision，recall。f1
                :param y_true:
                :param probas_pred:
                :return:
                '''
                precisions, recalls, thresholds = precision_recall_curve(
                    y_true, probas_pred)

                f1s = (2 * precisions * recalls) / (precisions + recalls)
                f1s[np.isnan(f1s)] = 0

                best_index = np.argmax(f1s)

                aupr = average_precision_score(y_true, probas_pred)

                return precisions[best_index], recalls[best_index], f1s[best_index], aupr


            ##trace level
            trace_p, trace_r, trace_f1, trace_aupr = cal_best_PRF(dataset.case_target, trace_level_abnormal_scores)
            print("trace")
            print(trace_p, trace_r, trace_f1, trace_aupr)

            ##event level
            eventTemp = dataset.binary_targets.sum(2).flatten()
            eventTemp[eventTemp > 1] = 1
            event_p, event_r, event_f1, event_aupr = cal_best_PRF(eventTemp, event_level_abnormal_scores.flatten())
            print("event")
            print(event_p, event_r, event_f1, event_aupr)

            ##attr level
            attr_p, attr_r, attr_f1, attr_aupr = cal_best_PRF(dataset.binary_targets.flatten(),
                                                              attr_level_abnormal_scores.flatten())
            print("attr")
            print(attr_p, attr_r, attr_f1, attr_aupr)

            self.metrics= pd.DataFrame([{'trace_p': trace_p, "trace_r": trace_r,'trace_f1':trace_f1,'trace_aupr':trace_aupr,
                                 'event_p': event_p, "event_r": event_r, 'event_f1': event_f1, 'event_aupr': event_aupr,
                                 'attr_p': attr_p, "attr_r": attr_r, 'attr_f1': attr_f1, 'attr_aupr': attr_aupr}])