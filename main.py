import argparse
import os.path as osp
import wandb
import torch
import torch.nn.functional as F

from torch_geometric.datasets import Entities
from models.encoders import CompGCN, RGCN

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["AIFB", "AM"])
parser.add_argument("--compgcn", action="store_true")       # use CompGCN or R-GCN
parser.add_argument("--wandb", action="store_true")         # use wandb for tracking
parser.add_argument("--lr", type=float, default=0.0003)     # learning rate
parser.add_argument("--drop", type=float, default=0.0)      # dropout
parser.add_argument("--epochs", type=int, default=2000)     # number of epochs
parser.add_argument("--dim", type=int, default=2)           # dimensionality of one-hot encoded features
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--wd", type=float, default=0.0005)
parser.add_argument("--short_cut", action="store_true")     # residual connections
parser.add_argument("--msg_func", type=str, default="distmult")
parser.add_argument("--aggr_func", type=str, default="add", choices=["add", "mean", "max", "min", "pna"])
parser.add_argument("--layer_norm", action="store_true")
parser.add_argument("--compgcn_no_dir", action="store_true")     # ablation: no direction weights in CompGCN
parser.add_argument("--compgcn_no_relupd", action="store_true")  # ablation: no relation update in CompGCN
parser.add_argument("--rgcn_fast", action="store_true")          # fast version of RGCN, can be always turned on
parser.add_argument("--no_norm", action="store_true")            # ablation: do not normalize the adj in CompGCN
parser.add_argument("--drop_bias", action="store_true")          # drop bias for R-GCN
parser.add_argument("--mod_rgcn", action="store_true")           # modified R-GCN with additional MLP over features
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# seeding everything
torch.manual_seed(args.seed)


def main():
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Entities")
    dataset = Entities(path, args.dataset)
    data = dataset[0]

    # initialization of node features to the same one-hot vector
    data.x = torch.zeros((data.num_nodes, args.dim))
    data.x[:, 0] = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") if args.dataset == "AM" else device

    if args.compgcn:
        model = CompGCN(
            dims=[args.dim] * args.num_layers,
            num_relations=dataset.num_relations,
            num_classes=dataset.num_classes,
            message_func=args.msg_func,
            aggregate_func=args.aggr_func,
            layer_norm=args.layer_norm,
            short_cut=args.short_cut,
            use_dir_weight=not args.compgcn_no_dir,
            use_rel_update=not args.compgcn_no_relupd,
            use_norm=not args.no_norm,
        )
    else:
        model = RGCN(
            dims=[args.dim] * args.num_layers,
            num_relations=dataset.num_relations,
            num_classes=dataset.num_classes,
            dropout=args.drop,
            short_cut=args.short_cut,
            fast=args.rgcn_fast,
            aggr=args.aggr_func,
            drop_bias=args.drop_bias,
            mod=args.mod_rgcn,
        )

    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    num_params = sum(p.numel() for p in model.parameters())
    args.num_params = num_params
    print(f"Number of parameters: {num_params}")

    print(f"Training nodes: {len(data.train_idx)}")
    print(f"Test nodes: {len(data.test_idx)}")

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_type)
        loss = F.nll_loss(out[data.train_idx], data.train_y)
        loss.backward()
        optimizer.step()
        return loss.item()


    @torch.no_grad()
    def test():
        model.eval()
        out = model(data.x, data.edge_index, data.edge_type)
        pred = out.argmax(dim=-1)
        train_acc = pred[data.train_idx].eq(data.train_y).to(torch.float).mean()
        test_acc = pred[data.test_idx].eq(data.test_y).to(torch.float).mean()
        return train_acc.item(), test_acc.item()

    max_train, max_test = 0.0, 0.0

    if args.wandb:
        run = wandb.init(project="RWL")
        wandb.config.update(vars(args))

    for epoch in range(1, args.epochs):
        loss = train()
        train_acc, test_acc = test()
        max_train = max(max_train, train_acc)
        max_test = max(max_test, test_acc)
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} "
            f"Test: {test_acc:.4f}"
        )
        if args.wandb:
            wandb.log({"loss": loss, "train_acc": train_acc, "test_acc": test_acc})


if __name__ == "__main__":
    main()
