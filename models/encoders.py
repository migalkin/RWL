from collections.abc import Sequence

import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import FastRGCNConv, RGCNConv

from .compgcn_layer import CompGCNConv
from .mod_rgcn_layer import ModifiedRGCNConv, ModifiedFastRGCNConv


class CompGCN(nn.Module):

    def __init__(self, dims, num_relations, num_classes,
                 message_func=None, aggregate_func=None, layer_norm=False,
                 short_cut=False, activation="relu", use_dir_weight=True, use_rel_update=True, use_norm=True):
        super(CompGCN, self).__init__()

        if not isinstance(dims, Sequence):
            dims = [dims]
        self.dims = list(dims)
        self.short_cut = short_cut
        self.num_relations = num_relations

        self.relation_embs = torch.nn.Parameter(torch.empty(num_relations, dims[0]))
        nn.init.xavier_uniform_(self.relation_embs)

        layer_type = CompGCNConv

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer_type(
                input_dim=self.dims[i],
                output_dim=self.dims[i + 1],
                message_func=message_func,
                aggregate_func=aggregate_func,
                activation=activation,
                layer_norm=layer_norm,
                use_dir_weight=use_dir_weight,
                use_rel_update=use_rel_update,
                use_norm=use_norm,
                num_relations=num_relations,
            ))
        self.layers.append(layer_type(
            input_dim=self.dims[-1],
            output_dim=num_classes,
            message_func=message_func,
            aggregate_func=aggregate_func,
            activation=activation,
            layer_norm=layer_norm,
            use_dir_weight=use_dir_weight,
            use_rel_update=use_rel_update,
            use_norm=use_norm,
            num_relations=num_relations,
        ))


    def forward(self, x, edge_index, edge_type):
        """
        Compute the node representations and the graph representation(s).
        """
        hiddens = []
        layer_input = x

        relation_embs = self.relation_embs

        for i, layer in enumerate(self.layers):
            hidden, relation_embs = layer(layer_input, edge_index, edge_type, relation_embs)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        node_feature = hiddens[-1]

        return F.log_softmax(node_feature, dim=1)

class RGCN(torch.nn.Module):
    def __init__(self, dims, num_relations, dropout, num_classes, short_cut=False,
                 fast=False, aggr="mean", drop_bias=False, mod=False):
        super().__init__()

        self.dims = dims
        self.short_cut = short_cut
        if mod:
            layer_type = ModifiedRGCNConv if not fast else ModifiedFastRGCNConv
            self.need_relu = False
        else:
            layer_type = RGCNConv if not fast else FastRGCNConv
            self.need_relu = True

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layer_type(
                    in_channels=self.dims[i],
                    out_channels=self.dims[i + 1],
                    num_relations=num_relations,
                    num_bases=None,
                    num_blocks=None,
                    aggr=aggr,
                    bias=not drop_bias,
                )
            )
        self.layers.append(
            layer_type(
                in_channels=self.dims[-1],
                out_channels=num_classes,
                num_relations=num_relations,
                num_bases=None,
                num_blocks=None,
                aggr=aggr,
                bias=not drop_bias
            )
        )

        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, x, edge_index, edge_type):
        hiddens = []
        layer_input = x

        for i, layer in enumerate(self.layers):
                hidden = layer(layer_input, edge_index, edge_type)
                if self.need_relu:
                    # Default R-GCN does not have a non-linearity in the output, need a manual relu
                    # no activation after final layer
                    if i != len(self.layers) - 1:
                        hidden = F.relu(hidden)
                    hidden = F.relu(hidden)

                if self.short_cut and hidden.shape == layer_input.shape:
                    hidden += layer_input

                hiddens.append(hidden)
                layer_input = hidden

        node_feature = hiddens[-1]

        return F.log_softmax(node_feature, dim=1)
