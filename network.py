import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import config

def detect_abnormal(t, message):
    assert not t.isinf().any(), f"inf: {message}"
    assert not t.isnan().any(), f"nan: {message}"

class Softmax_1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        x_max = x.amax(-1, keepdims = True)
        x_adj_exp = (x - x_max).exp()
        p = x_adj_exp / (torch.exp(-x_max) + x_adj_exp.sum(-1, keepdims = True))
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        p, = ctx.saved_tensors
        grad_x = p * (grad_output - (p * grad_output).sum(-1, keepdims = True))
        return grad_x


class GDMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, attn_pdrop, bias):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.attn_dropout = nn.Dropout(p = attn_pdrop)
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias = bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x, attn_bias, attn_bias_mul, key_padding_mask):
        r"""
            x[seq_len, batch_size, embed_dim]
            attn_bias[batch_size, num_heads, seq_len, seq_len] or None
            key_padding_mask[batch_size, seq_len] or None
                mask to exclude keys that are pads, where padding elements are indicated by 1s.
        """
        seq_len, batch_size, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"x dim {embed_dim} != {self.embed_dim}"

        x = self.in_proj(x) # [seq_len, batch_size, 3 * embed_dim]
        x = x.view(seq_len, batch_size * self.num_heads, 3 * self.head_dim).transpose(0, 1) # [batch_size * num_heads, seq_len, 3 * self.head_dim]
        q, k, v = x.split(self.head_dim, dim = -1)
        q = q * self.scaling
        # q, k, v: [batch_size * num_heads, seq_len, head_dim]

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # attn_weights [batch_size * num_heads, q_seq_len, k_seq_len]

        # assert attn_weights.max() < 50, f"a max: {attn_weights.max()}.{which}"
        # assert attn_bias.max() < 50, f"b max: {attn_bias.max()}.{which}"

        if attn_bias is not None:
            assert attn_bias.size() == (batch_size, self.num_heads, seq_len, seq_len)
            attn_weights += attn_bias.reshape(batch_size * self.num_heads, seq_len, seq_len)

        attn_weights = self.attn_dropout(attn_weights)

        if key_padding_mask is not None:
            assert key_padding_mask.size() == (batch_size, seq_len)
            # don't attend to padding symbols
            attn_weights = attn_weights.view(batch_size, self.num_heads, seq_len, seq_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
            attn_weights = attn_weights.view(batch_size * self.num_heads, seq_len, seq_len)
            if attn_bias_mul is not None:
                assert attn_bias_mul.size() == (batch_size, self.num_heads, seq_len, seq_len)
                attn_bias_mul = attn_bias_mul.masked_fill(key_padding_mask[:, None, None, :], 0.)
                attn_bias_mul = attn_bias_mul.view(batch_size * self.num_heads, seq_len, seq_len)

        attn_weights = Softmax_1.apply(attn_weights)
        if attn_bias_mul is not None:
            attn_weights = attn_weights * attn_bias_mul

        attn = torch.bmm(attn_weights, v) # [batch_size * num_heads, q_seq_len, head_dim]
        assert attn.size() == (batch_size * self.num_heads, seq_len, self.head_dim)

        attn = attn.transpose(0, 1).reshape(seq_len, batch_size, embed_dim)
        attn = self.out_proj(attn)

        return attn


class GraphormerGDGraphEncoderLayer(nn.Module):

    def __init__(self, embedding_dim, ffn_embedding_dim, num_attention_heads, resid_pdrop, attn_pdrop):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads

        self.pre_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.self_attn = GDMultiheadAttention(
            embed_dim = embedding_dim,
            num_heads = num_attention_heads,
            attn_pdrop = attn_pdrop,
            bias = True,
        )

        self.pre_ffn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.gelu = nn.GELU()

    def forward(self, x, attn_bias, attn_bias_mul, key_padding_mask):
        # x: [seq_len, batch_size, embedding_dim]
        # attn_bias, [batch_size, num_heads, seq_len, seq_len]
        # key_padding_mask[batch_size, seq_len] bool

        identity = x
        x = self.pre_attn_layer_norm(x)
        x = self.self_attn(x = x, attn_bias = attn_bias, attn_bias_mul = attn_bias_mul, key_padding_mask = key_padding_mask)
        x = self.resid_dropout(x)
        x = identity + x

        identity = x
        x = self.pre_ffn_layer_norm(x)
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.resid_dropout(x)
        x = identity + x
        return x


class GraphNodeFeature(nn.Module):

    def __init__(self, x_long_dim, x_real_dim, hidden_dim):
        super().__init__()

        self.long_encoder = nn.Embedding(config._node_embeddings, hidden_dim, padding_idx = 0)
        self.real_encoder = nn.Linear(x_real_dim, hidden_dim)
        self.degree_encoder = nn.Embedding(config._max_num_nodes, hidden_dim, padding_idx = 0)
        self.graph_token = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, batch_data):
        x_long, x_real, degree = (
            batch_data['x_long'],
            batch_data['x_real'],
            batch_data['degree'],
        )
        n_graph, n_node = x_long.size()[:2]

        x_long = self.long_encoder(x_long).mean(dim = -2) # [n_graph, n_node, hidden_dim]
        x_real = self.real_encoder(x_real) # [n_graph, n_node, hidden_dim]
        degree = self.degree_encoder(degree) # [n_graph, n_node, hidden_dim]

        node_feature = torch.mean(torch.stack([x_long, x_real, degree]), dim = 0)
        graph_token_feature = self.graph_token[None, None, :].expand(n_graph, -1, -1)
        node_feature = torch.cat([graph_token_feature, node_feature], dim = 1) # [n_graph, n_node + 1, hidden_dim]

        return node_feature


class FcResBlock(nn.Module):
    
    def __init__(self, hidden_dim, resid_pdrop):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = x

        x = self.fc2(self.gelu(self.fc1(self.norm(x))))
        x = self.resid_dropout(x)
        x += identity

        return x


class GraphAttnBias(nn.Module):
    r"""
        input:
            spatial_pos,
            edge_long,
            edge_long_path,
            action_pos,
            res_pos,
            node_type_edge,
        output:
            graph_attn_bias[n_layer, n_graph, n_head, n_node, n_node]
    """

    def __init__(self, num_heads, num_layers, e_dim, resid_pdrop, hidden_dim = 32, num_res_blocks = 6):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.e_dim = e_dim
        self.hidden_dim = hidden_dim

        self.spatial_pos_encoder = nn.Embedding(config._max_num_nodes, num_layers * hidden_dim, padding_idx = 0)
        self.action_pos_encoder = nn.Embedding(config._action_embeddings, num_layers * hidden_dim, padding_idx = 0)
        self.edge_encoder = nn.Embedding(config._edge_embeddings, num_layers * hidden_dim, padding_idx = 0)
        self.res_encoder = nn.Linear(1, num_layers * hidden_dim)
        self.node_type_edge_encoder = nn.Embedding(2 * config._max_atom_types, num_layers * hidden_dim, padding_idx = 0)

        self.layers = nn.ModuleList([FcResBlock(hidden_dim, resid_pdrop) for _ in range(num_res_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_heads)

        self.t = nn.Parameter(torch.randn(3, num_layers, num_heads))

        self.gelu = nn.GELU()

    def forward(self, batch_data):
        spatial_pos,  edge_long, action_pos, res_pos, node_type_edge = (
            batch_data['spatial_pos'],
            batch_data['edge_long'],
            batch_data['action_pos'],
            batch_data['res_pos'],
            batch_data['node_type_edge'],
        )
        # spatial_pos[n_graph, n_node, n_node] shortest_path_result
        # edge_long[n_graph, n_node, n_node, e_dim] 0 padding, feature from 2, 3, 4
        # action_pos[n_graph, n_node, n_node] 0 padding, feature from 1, 2, 3...
        # res_pos[n_graph, n_node, n_node]
        # node_type_edge[n_graph, n_node, n_node, 2] node type for each edge, 0-padding
        n_graph, n_node = spatial_pos.size()[:2]
        n_head, n_layer, hidden_dim = self.num_heads, self.num_layers, self.hidden_dim
        gnn_mask = batch_data['spatial_pos'].gt(0) # [n_graph, n_node, n_node] mask

        spatial_pos = self.spatial_pos_encoder(spatial_pos[gnn_mask]).reshape(-1, hidden_dim)
        action_pos = self.action_pos_encoder(action_pos[gnn_mask]).reshape(-1, hidden_dim)
        edge_long = self.edge_encoder(edge_long[gnn_mask]).mean(-2).reshape(-1, hidden_dim)
        res_pos = self.res_encoder(res_pos[gnn_mask].unsqueeze(-1)).reshape(-1, hidden_dim)
        node_type_edge = self.node_type_edge_encoder(node_type_edge[gnn_mask]).mean(-2).reshape(-1, hidden_dim)

        x = torch.mean(torch.stack([spatial_pos, action_pos, edge_long, res_pos, node_type_edge]), dim = 0)
        # x[n_graph * n_node * n_node * n_layer, hidden_dim]

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        z = x.new_zeros(n_graph, n_node, n_node, n_layer, n_head)
        z[gnn_mask] = x.view(-1, n_layer, n_head)
        z = z.permute(3, 0, 4, 1, 2)
        # z[n_layer, n_graph, n_head, n_node, n_node]

        out = z.new_zeros(n_layer, n_graph, n_head, n_node + 1, n_node + 1)
        out[:, :, :, 1:, 1:] = z
        out[:, :, :, 0, 0] = self.t[0][:, None, :]
        out[:, :, :, 0, 1:] = self.t[1][:, None, :, None]
        out[:, :, :, 1:, 0] = self.t[2][:, None, :, None]

        return out


class GraphormerGDGraphEncoder(nn.Module):
    def __init__(self, x_long_dim, x_real_dim, e_dim, num_encoder_layers, embedding_dim, ffn_embedding_dim, num_attention_heads,
                 resid_pdrop, attn_pdrop):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_attention_heads = num_attention_heads

        self.graph_node_feature = GraphNodeFeature(x_long_dim = x_long_dim, x_real_dim = x_real_dim, hidden_dim = embedding_dim)

        self.graph_attn_bias = GraphAttnBias(
            num_heads = num_attention_heads,
            num_layers = num_encoder_layers * 2,
            e_dim = e_dim,
            resid_pdrop = resid_pdrop,
        )

        self.layers = nn.ModuleList([GraphormerGDGraphEncoderLayer(
            embedding_dim = embedding_dim,
            ffn_embedding_dim = ffn_embedding_dim,
            num_attention_heads = num_attention_heads,
            resid_pdrop = resid_pdrop,
            attn_pdrop = attn_pdrop,
        ) for _ in range(num_encoder_layers)])

    def forward(self, batch_data):
        r"""
            x_long[n_graph, n_node, x_long_dim] long: should have 0-padding
            degree[n_graph, n_node] long: should have 0-padding
            spatial_pos[n_graph, n_node, n_node] long: 0-padding and shift + 1
            edge_long[n_graph, n_node, n_node, e_dim] 0 padding, feature from 2, 3, 4, ...
            edge_long_path[n_graph, n_node, n_node, max_dist, e_dim] 0 padding, feature from 2, 3, 4, ...
            action_pos[n_graph, n_node, n_node] 0 padding, feature from 1, 2, 3...
        """

        n_graph, n_node = batch_data['x_long'].size()[:2]

        # compute padding mask. This is needed for multi-head attention
        key_padding_mask = batch_data['x_long'][:, :, 0].eq(0) # [n_graph, n_node]
        key_padding_mask_cls = key_padding_mask.new_zeros(n_graph, 1)
        key_padding_mask = torch.cat([key_padding_mask_cls, key_padding_mask], dim = 1) # [n_graph, n_node + 1]

        attn_bias = self.graph_attn_bias(batch_data)
        # [num_layers, n_graph, num_heads, n_node + 1, n_node + 1]

        x = self.graph_node_feature(batch_data) # x[n_graph, n_node + 1, hidden_dim]
        x = x.transpose(0, 1) # [seq_len, batch_size, hidden_dim]

        for i, layer in enumerate(self.layers):
            x = layer(
                x = x,
                attn_bias = attn_bias[i * 2],
                attn_bias_mul = attn_bias[i * 2 + 1],
                key_padding_mask = key_padding_mask,
            )

        return x

class Network(nn.Module):

    def __init__(self, x_long_dim, x_real_dim, e_dim, num_encoder_layers, embedding_dim, ffn_embedding_dim, num_attention_heads,
                 resid_pdrop, attn_pdrop,
                 num_obk_actions, num_rbk_actions, num_exn_actions, num_global_neg_hs_actions):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_obk_actions = num_obk_actions
        self.num_rbk_actions = num_rbk_actions
        self.num_exn_actions = num_exn_actions
        self.num_global_neg_hs_actions = num_global_neg_hs_actions

        self.encoder = GraphormerGDGraphEncoder(
            x_long_dim = x_long_dim,
            x_real_dim = x_real_dim,
            e_dim = e_dim,
            num_encoder_layers = num_encoder_layers,
            embedding_dim = embedding_dim,
            ffn_embedding_dim = ffn_embedding_dim,
            num_attention_heads = num_attention_heads,
            resid_pdrop = resid_pdrop,
            attn_pdrop = attn_pdrop,
        )
        self.post_encoder_layer_norm = nn.LayerNorm(embedding_dim)

        self.obk_proj = nn.Linear(12 * embedding_dim, num_obk_actions)

        self.rbk_proj = nn.Linear(6 * embedding_dim, num_rbk_actions)

        self.exn_proj = nn.Linear(17 * embedding_dim, num_exn_actions)

        self.global_neg_hs_proj = nn.Linear(4 * embedding_dim, num_global_neg_hs_actions)

        self.halt_proj = nn.Linear(4 * embedding_dim, 1)
        
        self.gelu = nn.GELU()

    def forward(self, batch_data):
        r"""
            batch_data is a $dict$ holding all fields related to the molecule

            $n_graph$ is the number of graphs contained in the batch

            $n_node$ is the maximum number of nodes of all graphs. It is possible some graph has less nodes than $n_node$,
            usually its tensor will be padded with zeros (see details in the followin)

            $n_obk$, $n_rbk$, $exn$ is the maximum number of one-bond break/ring-bond break/excision actions per each molecule

            embeddin values are guarantee to be > 0 and small graphs have 0-padding to the remaining matrix
            This can be useful, e.g., x_long.gt(0).any(-1) or x_long[:, :, 0].gt(0) can serve as node mask

            x_long[n_graph][n_node][x_long_dim], long: integer value node features. [x_long_dim] is a one-dimensional encoding of multi-category embeddins
            x_real[n_graph][n_node][x_real_dim], float: real value node features
            edge_long[n_graph][n_node][n_node][e_dim], long: edge features of each pair of nodes, 0-padding for non-exsisting edge between any pair of nodes
            spatial_pos[n_graph][n_node][n_node], long: distance matrix, 0-padded
            res_pos[n_graph][n_node][n_node], float: resistance distance matrix, 0-padded
            action_pos[n_graph][n_node][n_node], long: action matrix, 0-padded
            degree[n_graph][n_node], long: degree of each node shifted by +1 to make it 0-padded
            obk_edge[n_graph][n_obk][2], long
            obk_comp[n_graph][n_obk][2], bool
            rbk_edge[n_graph][n_rbk][2], long
            exn_quad[n_graph][n_exn][4], long
            exn_comp[n_graph][n_exn][3], bool
            node_type_edge[n_graph][n_node][n_node][2], long

        """
        (
            obk_edge, # [n_graph, n_obk, 2] long, 0-padding
            obk_comp, # [n_graph, n_obk, 2, n_node] bool
            rbk_edge, # [n_graph, n_rbk, 2] long, 0-padding
            exn_quad, # [n_graph, n_exn, 4] long, s-u~v-t, 0-padding
            exn_comp, # [n_graph, n_exn, 3, n_node] bool, comp_s, comp_u_v, comp_t
            action_mask, # [n_graph, num_tot_actions] bool
        ) = (
            batch_data['obk_edge'],
            batch_data['obk_comp'],
            batch_data['rbk_edge'],
            batch_data['exn_quad'],
            batch_data['exn_comp'],
            batch_data['action_mask'],
        )

        n_graph, n_obk = obk_edge.size()[:2]
        n_rbk = rbk_edge.size(1)
        n_exn = exn_quad.size(1)
        
        x = self.encoder(batch_data).transpose(0, 1)

        x = self.post_encoder_layer_norm(x) # x[n_graph, n_node + 1, hidden_dim]
        x = self.gelu(x)

        """ compute global feature """
        x_virtual = x[:, 0, :] # [n_graph, hidden_dim]
        x = x[:, 1:, :] # [n_graph, n_node, hidden_dim]
        node_mask = batch_data['x_long'][:, :, 0].gt(0) # [n_graph, n_node]
        x_sum = (x * node_mask.unsqueeze(-1)).sum(-2) # [n_graph, hidden_dim]
        x_mean = x_sum / node_mask.sum(-1, keepdims = True) # [n_graph, hidden_dim]
        x_max = x.masked_fill(~node_mask.unsqueeze(-1), -100.0).amax(-2) # [n_graph, hidden_dim]
        x_global = torch.cat([
            x_virtual[:, None, None, :],
            x_sum[:, None, None, :] * 0.01,
            x_mean[:, None, None, :],
            x_max[:, None, None, :],
        ], dim = -2) # [n_graph, 1, 4, hidden_dim]

        """ compute logit for one-bond breakage (obk) """
        # input[n_graph, n_obk, n_node, hidden_dim]
        # index[n_graph, n_obk, 2, hidden_dim]
        obk_node_feature = torch.gather(
            input = x.unsqueeze(1).expand(-1, n_obk, -1, -1), # [n_graph, n_obk, n_node, hidden_dim]
            dim = 2,
            index = obk_edge.unsqueeze(-1).expand(-1, -1, -1, self.embedding_dim), # [n_graph, n_obk, 2, hidden_dim]
        ) # [n_graph, n_obk, 2, hidden_dim]

        # x[:, None, None, :, :] -> x[n_graph, 1, 1, n_node, hidden_dim]
        # obk_comp[:, :, :, :, None] -> obk_comp[n_graph, n_obk, 2, n_node, 1]
        obk_comp_sum = (x[:, None, None, :, :] * obk_comp[:, :, :, :, None]).sum(-2) # obk_comp_sum[n_graph, n_obk, 2, hidden_dim]
        obk_comp_mean = obk_comp_sum / obk_comp.sum(-1, keepdims = True).clamp_(min = 1)
        obk_comp_max = x[:, None, None, :, :].expand(-1, n_obk, 2, -1, -1).masked_fill(~obk_comp.unsqueeze(-1), -100.0).amax(-2)

        obk_feature = torch.cat(
            [
                obk_node_feature,
                obk_comp_sum * 0.01,
                obk_comp_mean,
                obk_comp_max,
                x_global.expand(-1, n_obk, -1, -1)
            ],
            dim = -2,
        ).flatten(-2) # [n_graph, n_obk, 12 * hidden_dim]
        obk_logit = self.obk_proj(obk_feature) # [n_graph, n_obk, num_obk_actions]
        
        """ compute logit for ring-bond breakage (rbk) """
        rbk_node_feature = torch.gather(
            input = x.unsqueeze(1).expand(-1, n_rbk, -1, -1), # [n_graph, n_rbk, n_node, hidden_dim]
            dim = 2,
            index = rbk_edge.unsqueeze(-1).expand(-1, -1, -1, self.embedding_dim), # [n_graph, n_rbk, 2, hidden_dim]
        ) # [n_graph, n_rbk, 2, hidden_dim]
        rbk_feature = torch.cat(
            [rbk_node_feature, x_global.expand(-1, n_rbk, -1, -1)],
            dim = -2,
        ).flatten(-2) # [n_graph, n_rbk, 6 * hidden_dim]
        rbk_logit = self.rbk_proj(rbk_feature) # [n_graph, n_rbk, num_rbk_actions]

        """ compute logit for excision (exn) """
        exn_node_feature = torch.gather(
            input = x.unsqueeze(1).expand(-1, n_exn, -1, -1), # [n_graph, n_exn, n_node, hidden_dim]
            dim = 2,
            index = exn_quad.unsqueeze(-1).expand(-1, -1, -1, self.embedding_dim), # [n_graph, n_exn, 4, hidden_dim]
        ) # [n_graph, n_exn, 4, hidden_dim]

        # [n_graph, n_exn, 3, n_node, hidden_dim] -> [n_graph, n_exn, 3, hidden_dim]
        exn_comp_sum = (x[:, None, None, :, :] * exn_comp[:, :, :, :, None]).sum(-2)
        exn_comp_mean = exn_comp_sum / exn_comp.sum(-1, keepdims = True).clamp_(min = 1)
        exn_comp_max = x[:, None, None, :, :].expand(-1, n_exn, 3, -1, -1).masked_fill(~exn_comp.unsqueeze(-1), -100.0).amax(-2)
        exn_comp_feature = torch.cat([
            exn_comp_sum * 0.01,
            exn_comp_mean,
            exn_comp_max,
        ], -1) # [n_graph, n_exn, 3, 3 * hidden_dim]

        # to keep invariance of s-u and t-v, duplicate the data the inverse direction
        exn_feature = torch.cat([
            torch.cat([
                exn_node_feature.flatten(2), # [n_graph, n_exn, 4 * hidden_dim]
                exn_comp_feature.flatten(2), # [n_graph, n_exn, 3 * 3 * hidden_dim]
                x_global.expand(-1, n_exn, -1, -1).flatten(2), # [n_graph, n_exn, 4 * hidden_dim]
            ], dim = 2),
            torch.cat([
                exn_node_feature.flip(2).flatten(2), # [n_graph, n_exn, 4 * hidden_dim]
                exn_comp_feature.flip(2).flatten(2), # [n_graph, n_exn, 3 * 3 * hidden_dim]
                x_global.expand(-1, n_exn, -1, -1).flatten(2), # [n_graph, n_exn, 4 * hidden_dim]
            ], dim = 2),
        ], dim = 0) # [2 * n_graph, n_exn, 17 * hidden_dim]
        exn_logit = self.exn_proj(exn_feature) # [2 * n_graph, n_exn, num_exn_actions]
        exn_logit = exn_logit[:n_graph] + exn_logit[n_graph:] # [n_graph, n_exn, num_exn_actions]

        """ compute logit for global_neg_hs """
        global_neg_hs_logit = self.global_neg_hs_proj(x_global.flatten(1)) # [n_graph, num_global_neg_hs_actions]

        """ compute logit for halt """
        halt_logit = self.halt_proj(x_global.flatten(1)) # [n_graph, 1]

        """ aggregate all """
        logit = torch.cat([
            obk_logit.flatten(1),
            rbk_logit.flatten(1),
            exn_logit.flatten(1),
            global_neg_hs_logit.flatten(1),
            halt_logit.flatten(1),
        ], dim = -1)
        logit[~action_mask] = float('-inf')
        logit = torch.log_softmax(logit, -1)

        return logit

if __name__ == '__main__':
    # from torch.autograd import gradcheck
    # test = gradcheck(Softmax_1.apply, (torch.randn(10, 20, 30, requires_grad = True).double(),))
    # print(test)
    # from IPython import embed; embed()
    pass
