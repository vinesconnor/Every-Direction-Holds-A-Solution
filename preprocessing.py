from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
import numpy as np
import json
from sklearn.decomposition import PCA
import csv

def load_data(task: str, n_comps: int):
    if task == 'chem':
        return load_chem_data(n_comps)
    else:
        return load_drug_data(task, n_comps)

def load_chem_data(n_comps: int):
    with open('data/Chem/train.csv', 'r') as f:
        train_feats = np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:])
    with open('data/Chem/train.csv', 'r') as f:
        train_labels = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])

    with open('data/Chem/val_id.csv', 'r') as f:
        iid_val_feats = np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:])
    with open('data/Chem/val_id.csv', 'r') as f:
        iid_val_labels = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])

    with open('data/Chem/val_ood.csv', 'r') as f:
        ood_val_feats = np.float32(np.array([line.strip().split(',')[4:] for line in f])[1:])
    with open('data/Chem/val_ood.csv', 'r') as f:
        ood_val_labels = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])

    with open('data/Chem/test_id.csv', 'r') as f:
        iid_test_feats = np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:])
    with open('data/Chem/test_id.csv', 'r') as f:
        iid_test_labels = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])

    with open('data/Chem/test_ood.csv', 'r') as f:
        ood_test_feats = np.float32(np.array([line.strip().split(',')[4:] for line in f])[1:])
    with open('data/Chem/test_ood.csv', 'r') as f:
        ood_test_labels = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])

    train_feats, mu, sigma = normalize(train_feats)
    iid_val_feats = (iid_val_feats - mu) / sigma
    ood_val_feats = (ood_val_feats - mu) / sigma
    iid_test_feats = (iid_test_feats - mu) / sigma
    ood_test_feats = (ood_test_feats - mu) / sigma

    pca = PCA(n_components=n_comps)
    train_feats = pca.fit_transform(train_feats)
    iid_val_feats = pca.transform(iid_val_feats)
    ood_val_feats = pca.transform(ood_val_feats)
    iid_test_feats = pca.transform(iid_test_feats)
    ood_test_feats = pca.transform(ood_test_feats)

    return {
        'train': (train_feats, train_labels),
        'iid_val': (iid_val_feats, iid_val_labels),
        'ood_val': (ood_val_feats, ood_val_labels),
        'iid_test': (iid_test_feats, iid_test_labels),
        'ood_test': (ood_test_feats, ood_test_labels)
    }

def load_drug_data(task: str, n_comps: int):
    """
    Args:
        task - ic50, ec50, potency
        n_comps - num PCA components wanted
    Returns:
        Data dictionary of all splits of PCA'd data
    """
    split = load_json(task)
    train = split['train']
    iid_val = split['iid_val']
    ood_val = split['ood_val']
    iid_test = split['iid_test']
    ood_test = split['ood_test']

    train_feats = convert_data_to_feats(train)
    train_feats, mu, sigma = normalize(train_feats)
    train_labels = np.array([entry['cls_label'] for entry in train])

    iid_val_feats = convert_data_to_feats(iid_val)
    iid_val_feats = (iid_val_feats - mu) / sigma
    iid_val_labels = np.array([entry['cls_label'] for entry in iid_val])

    ood_val_feats = convert_data_to_feats(ood_val)
    ood_val_feats = (ood_val_feats - mu) / sigma
    ood_val_labels = np.array([entry['cls_label'] for entry in ood_val])

    iid_test_feats = convert_data_to_feats(iid_test)
    iid_test_feats = (iid_test_feats - mu) / sigma
    iid_test_labels = np.array([entry['cls_label'] for entry in iid_test])

    ood_test_feats = convert_data_to_feats(ood_test)
    ood_test_feats = (ood_test_feats - mu) / sigma
    ood_test_labels = np.array([entry['cls_label'] for entry in ood_test])

    pca = PCA(n_components=n_comps)
    train_feats = pca.fit_transform(train_feats)
    iid_val_feats = pca.transform(iid_val_feats)
    ood_val_feats = pca.transform(ood_val_feats)
    iid_test_feats = pca.transform(iid_test_feats)
    ood_test_feats = pca.transform(ood_test_feats)

    return {
        'train': (train_feats, train_labels),
        'iid_val': (iid_val_feats, iid_val_labels),
        'ood_val': (ood_val_feats, ood_val_labels),
        'iid_test': (iid_test_feats, iid_test_labels),
        'ood_test': (ood_test_feats, ood_test_labels)
    }

def load_json(task: str):
    path = f'data/DrugOOD/sbap_core_{task}_protein.json'
    with open(path, 'r') as f:
        data = json.load(f)
    return data['split']

def convert_data_to_feats(data):
    smiles_feats = []
    for entry in data:
        smiles = entry['smiles']
        smiles_feats.append(smiles_to_ecfp(smiles))
    smiles_feats = np.array(smiles_feats)
    return smiles_feats

def smiles_to_ecfp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    array = np.zeros((0,), dtype=int)
    DataStructs.ConvertToNumpyArray(ecfp, array)
    return array

# Normalize the training data and return mu, sigma
def normalize(X):
    mu = np.mean(X, 0, keepdims=True)
    sigma = np.std(X, 0, keepdims=True)
    #sigma = np.ones_like(mu)
    return (X-mu)/sigma, mu, sigma