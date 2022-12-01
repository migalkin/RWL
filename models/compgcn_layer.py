import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min


class CompGCNConv(MessagePassing):

    eps = 1e-6

    def __init__(self, input_dim, output_dim, message_func="distmult",
                 aggregate_func="pna", activation="relu", layer_norm=False,
                 use_rel_update=True, use_dir_weight=True, use_norm=True, num_relations=None):

        super(CompGCNConv, self).__init__(flow="target_to_source", aggr=aggregate_func if aggregate_func != "pna" else None)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.use_norm = use_norm
        self.num_relations = num_relations

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 12, output_dim)
        if self.message_func == "mlp":
            self.msg_mlp = nn.Sequential(
                nn.Linear(2 * input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim)
            )

        self.use_rel_update = use_rel_update
        self.use_dir_weight = use_dir_weight

        if self.use_rel_update:
            self.relation_update = nn.Linear(input_dim, input_dim)

        # CompGCN weight matrices
        if self.use_dir_weight:
            self.w_in = nn.Parameter(torch.empty(input_dim, output_dim))
            self.w_out = nn.Parameter(torch.empty(input_dim, output_dim))
            self.w_loop = nn.Parameter(torch.empty(input_dim, output_dim))
            nn.init.xavier_uniform_(self.w_in)
            nn.init.xavier_uniform_(self.w_out)
            nn.init.xavier_uniform_(self.w_loop)
        else:
            self.w = nn.Parameter(torch.empty(input_dim, output_dim))
            nn.init.xavier_uniform_(self.w)

        # layer-specific self-loop relation parameter
        self.loop_relation = nn.Parameter(torch.empty(1, input_dim))
        nn.init.xavier_uniform_(self.loop_relation)


    def forward(self, x, edge_index, edge_type, relation_embs):
        """
        CompGCN forward pass is the average of direct, inverse, and self-loop messages
        """

        # out graph -> the original graph without inverse edges
        edge_list = edge_index

        # in PyG Entities datasets, direct edges have even indices, inverse - odd
        if self.use_dir_weight:
            out_index = edge_list[:, edge_type % 2 == 0]
            out_type = edge_type[edge_type % 2 == 0]
            out_norm = self.compute_norm(out_index, x.shape[0]) if self.use_norm else torch.ones_like(out_type, dtype=torch.float)

            # in graph -> the graph with only inverse edges
            in_index = edge_list[:, edge_type % 2 == 1]
            in_type = edge_type[edge_type % 2 == 1]
            in_norm = self.compute_norm(in_index, x.shape[0]) if self.use_norm else torch.ones_like(in_type, dtype=torch.float)

            # self_loop graph -> the graph with only self-loop relation type
            node_in = node_out = torch.arange(x.shape[0], device=x.device)
            loop_index = torch.stack([node_in, node_out], dim=0)
            loop_type = torch.zeros(loop_index.shape[1], dtype=torch.long, device=x.device)

            # message passing
            out_update = self.propagate(x=x, edge_index=out_index, edge_type=out_type, relation_embs=relation_embs, relation_weight=self.w_out, edge_weight=out_norm)
            in_update = self.propagate(x=x, edge_index=in_index, edge_type=in_type, relation_embs=relation_embs, relation_weight=self.w_in, edge_weight=in_norm)
            loop_update = self.propagate(x=x, edge_index=loop_index, edge_type=loop_type, relation_embs=self.loop_relation, relation_weight=self.w_loop)

            output = (out_update + in_update + loop_update) / 3.0

        else:
            # add self-loops
            node_in = node_out = torch.arange(x.shape[0], device=x.device)
            loop_index = torch.stack([node_in, node_out], dim=0)
            edge_index = torch.cat([edge_index, loop_index], dim=-1)

            loop_type = torch.zeros(loop_index.shape[1], dtype=torch.long, device=x.device).fill_(len(relation_embs))
            edge_type = torch.cat([edge_type, loop_type], dim=-1)
            relation_embs = torch.cat([relation_embs, self.loop_relation], dim=0)

            norm = self.compute_norm(edge_index, num_ent=x.shape[0]) if self.use_norm else torch.ones_like(edge_type, dtype=torch.float)
            output = self.propagate(
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                relation_embs=relation_embs,
                relation_weight=self.w,
                edge_weight=norm)

        if self.use_rel_update:
            relation_embs = self.relation_update(relation_embs)

        return output, relation_embs

    def message(self, x_j, edge_type, relation_embs, relation_weight, edge_weight=None):

        edge_input = relation_embs[edge_type]

        if self.message_func == "transe":
            message = edge_input + x_j
        elif self.message_func == "distmult":
            message = edge_input * x_j
        elif self.message_func == "rotate":
            node_re, node_im = x_j.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "mlp":
            message = self.msg_mlp(torch.cat([x_j, edge_input], dim=-1))
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # message transformation: can be direction-wise or simple linear map
        message = torch.mm(message, relation_weight)

        return message if edge_weight is None else message * edge_weight.view(-1, 1)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):

        if self.aggregate_func != "pna":
            return super().aggregate(inputs=inputs, index=index, ptr=ptr, dim_size=dim_size)
        else:
            mean = scatter_mean(inputs, index, dim=0, dim_size=dim_size)
            sq_mean = scatter_mean(inputs ** 2, index, dim=0, dim_size=dim_size)
            max = scatter_max(inputs, index, dim=0, dim_size=dim_size)[0]
            min = scatter_min(inputs, index, dim=0, dim_size=dim_size)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)

            deg = degree(index, dim_size, dtype=inputs.dtype)
            scale = (deg+1).log()
            scale = scale / scale.mean()
            scales = torch.stack([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)

        return update

    def update(self, inputs):
        # in CompGCN, we just return updated states, no aggregation with inputs
        # update = update
        output = inputs if self.aggregate_func != "pna" else self.linear(inputs)
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    @staticmethod
    def compute_norm(edge_index, num_ent):
        """
        Re-normalization trick used by GCN-based architectures without attention.
        """
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()  # Identity matrix where we know all entities are there
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of
        # the edges, D = A + I
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0  # for numerical stability
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # Norm parameter D^{-0.5} *

        return norm
