import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

from torch.utils.data import DataLoader
from model import SeqTagger


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # TODO: implement main function
    #raise NotImplementedError
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTagger] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }    
    # TODO: crecate DataLoader for train / dev datasets
    DataLoaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 1, pin_memory = True, collate_fn = split_dataset.collate_fn)
        for split, split_dataset in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings,
        vocab.pad_id,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        datasets[TRAIN].num_classes,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    criterion = torch.nn.CrossEntropyLoss() 
    best_acc = 0.0
    model_path = args.ckpt_dir / "GRU_p2_test.pth"
    for epoch in epoch_pbar:
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        model.train()
        for data, data_len, labels in tqdm(DataLoaders[TRAIN]):
            # Train
            data, labels = data.to(args.device), labels.to(args.device)
            b, l = labels.shape
            optimizer.zero_grad() 
            
            outputs = model(data) 
            
            loss = criterion(outputs, labels.view(-1))
            loss.backward() 
            optimizer.step()
            
            _, train_pred = torch.max(outputs, 1) 
            train_pred = train_pred.reshape(b, l)
            acc = (train_pred.detach() == labels.detach()) | (labels.detach() == -100)
            train_acc += torch.all(acc, dim = 1).sum().item()
            train_loss += loss.item()
            
        # Val
        model.eval() 
        with torch.no_grad():
            for data, data_len, labels in tqdm(DataLoaders[DEV]):
                data, labels = data.to(args.device), labels.to(args.device)
                b, l = labels.shape
                outputs = model(data) 
                
                loss = criterion(outputs, labels.view(-1)) 
                
                _, val_pred = torch.max(outputs, 1) 
                val_pred = val_pred.reshape(b, l)
                acc = (val_pred.cpu() == labels.cpu()) | (labels.cpu()==-100)
                val_acc += torch.all(acc, dim = 1).sum().item() 
                val_loss += loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, args.num_epoch, train_acc/len(datasets[TRAIN]), train_loss/len(DataLoaders[TRAIN]), val_acc/len(datasets[DEV]), val_loss/len(DataLoaders[DEV])
            ))

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(datasets[DEV])))




def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)