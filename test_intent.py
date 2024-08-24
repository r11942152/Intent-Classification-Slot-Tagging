import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

from torch.utils.data import DataLoader
from tqdm import tqdm

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    DataLoader_A = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = 1, pin_memory = True, collate_fn = dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    model.to(args.device)
    # TODO: predict dataset
    ids = []
    intents = []
    model.eval() 
    with torch.no_grad():
        for data, id in tqdm(DataLoader_A):
            data = data.to(args.device)
            outputs = model(data) 
                
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().tolist()
            ids += id
            intents += [dataset.idx2label(pred) for pred in preds]
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as f:
        f.write('id,intent\n')
        for id, intent in zip(ids, intents):
            f.write('{},{}\n'.format(id, intent))
    print(f'write in {args.pred_file}')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True#,
        #default=".ckpt/intent/GRU_p1.pth"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

#python3 test_intent.py --test_file ./data/intent/test.json --ckpt_path ./ckpt/intent/GRU_p1.pth
