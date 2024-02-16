_max_num_nodes = 150
_max_tot_hs = 160
_max_atom_types = 128
_range_shift_hs = 30
_range_formal_charge = 7
_max_total_valence = 12


_node_long_dim = 9
_node_real_dim = 1
_meta_long_dim = 8
_meta_real_dim = 3

_x_long_dim = _node_long_dim + _meta_long_dim
_x_real_dim = _node_real_dim + _meta_real_dim
_e_dim = 3

_num_obk_actions = 4
_obk_shift = -2
_num_rbk_actions = 3
_rbk_shift = -1
_max_excision_group = 1
_num_exn_actions = 4
_exn_shift = -2
_max_global_neg_hs = 4

_node_embeddings = _max_atom_types + 10 + _range_shift_hs * 2 + _range_formal_charge * 2 + _max_total_valence + _max_tot_hs + 30
_edge_embeddings = 8
_action_embeddings = 50
