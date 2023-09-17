from utils.dataset import Dataset
from utils.dataset_syn import Dataset_syn
from sklearn.preprocessing import LabelEncoder




def read_client_data(dataset, idx, AP,common_attrV,unique_attrV):
    dataset_name = '{}_{}-{:.2f}.json.gz'.format(dataset, idx, float(AP))
    print("reading:" + dataset_name)
    encoders = {}
    for key,value in common_attrV.items():
        encoder = LabelEncoder()

        encoder.classes_ = common_attrV[key] + unique_attrV[key]
        encoders[key] = encoder

    dataset = Dataset(dataset_name,encoders, beta=0.005)
    return dataset

def read_client_data_forSYN(dataset, idx, AP):
    dataset_name = '{}_{}-{:.2f}.json.gz'.format(dataset, idx, float(AP))
    dataset_name_clean = '{}_{}-{:.2f}.json.gz'.format(dataset, idx, float(0.00))

    dataset = Dataset_syn(dataset_name)
    dataset_clean = Dataset_syn(dataset_name_clean)
    return dataset_clean , dataset



