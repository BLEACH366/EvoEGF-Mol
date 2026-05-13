from absl import logging

import numpy as np
from tqdm import trange,tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from core.config.config import Struct
from core.models.common import compose_context, ShiftedSoftplus
from core.models.pif_base import PIFBase
from core.models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from core.models.uni_transformer_edge import UniTransformerO2TwoUpdateGeneralBond
from core.utils.frag_part_filter_func import bfs_substructure_mask, get_batch_connectivity_matrix, get_batch_type_pmf_matrix
# from core.models.e3_transformer import E3_transformer

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RBF(nn.Module):
    def __init__(self, start, end, n_center):
        super().__init__()
        self.start = start
        self.end = end
        self.n_center = n_center
        self.centers = torch.linspace(start, end, n_center)
        self.width = (end - start) / n_center

    def forward(self, x):
        assert x.ndim >= 2
        out = (x - self.centers.to(x.device)) / self.width
        ret = torch.exp(-0.5 * out**2)
        return F.normalize(ret, dim=-1, p=1) * 2 - 1


class TimeEmbedLayer(nn.Module):
    def __init__(self, time_emb_mode, time_emb_dim):
        super().__init__()
        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim

        if self.time_emb_mode == "simple":
            assert self.time_emb_dim == 1
            self.time_emb = lambda x: x
        elif self.time_emb_mode == "sin":
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        elif self.time_emb_mode == "rbf":
            self.time_emb = RBF(0, 1, self.time_emb_dim)
        elif self.time_emb_mode == "rbfnn":
            self.time_emb = nn.Sequential(
                RBF(0, 1, self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        else:
            raise NotImplementedError

    def forward(self, t):
        return self.time_emb(t)


class PIF4SBDDScoreModel(PIFBase):
    def __init__(
        self,
        net_config,
        protein_atom_feature_dim,
        ligand_atom_feature_dim,
        device="cuda",
        condition_time=True,
        use_discrete_t=False,
        discrete_steps=1000,
        t_min=0.0001,
        node_indicator=True,
        time_emb_mode='simple',
        time_emb_dim=1,
        center_pos_mode='protein',
        pos_init_mode='zero',
        destination_prediction = False,
        sampling_strategy = "vanilla",
        use_random_mask = True,
        c_s0 = 1.0,
        c_s1 = 0.1,
        d_s1 = 0.01,
        pm = 0.0,  # specify the probability of mask
        pam = 0.0,  # specify the probability of atom mask
    ):
        super().__init__()
        net_config = Struct(**net_config)
        self.config = net_config

        if net_config.name == 'unio2net':
            self.unio2net = UniTransformerO2TwoUpdateGeneral(**net_config.todict())
        elif net_config.name ==  'unio2net_bond':
            self.unio2net = UniTransformerO2TwoUpdateGeneralBond(**net_config.todict())
        # elif net_config.name == 'e3_transformer':
        #     self.unio2net = E3_transformer({})
        else:
            raise NotImplementedError
        
        self.c_s0 = c_s0
        self.c_s1 = c_s1
        self.d_s1 = d_s1        

        self.use_random_mask = use_random_mask
        self.pm = pm
        self.pam = pam

        self.hidden_dim = net_config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        self.num_bond_classes = self.config.num_bond_classes

        self.node_indicator = node_indicator

        if self.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.center_pos_mode = center_pos_mode  # ['none', 'protein']

        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim
        if self.time_emb_dim > 0:
            # self.time_emb_layer = TimeEmbedLayer(self.time_emb_mode, self.time_emb_dim)
            self.time_emb_layer1 = TimeEmbedLayer(self.time_emb_mode, self.time_emb_dim)
            self.time_emb_layer2 = TimeEmbedLayer(self.time_emb_mode, self.time_emb_dim)
        # self.ligand_atom_emb = nn.Linear(
        #     ligand_atom_feature_dim + self.time_emb_dim, emb_dim
        # )
        self.ligand_atom_emb = nn.Linear(
            ligand_atom_feature_dim + 2*self.time_emb_dim, emb_dim
        )
        self.ligand_bond_emb = nn.Linear(self.num_bond_classes, self.hidden_dim)

        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )  # [hidden to 13]

        if net_config.bond_net_type == 'lin':
            bond_input_dim = self.hidden_dim
        self.bond_inference = nn.Sequential(
            nn.Linear(bond_input_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, self.num_bond_classes)
        )

        self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time
        self.use_discrete_t = use_discrete_t  # whether to use discrete t
        self.discrete_steps = discrete_steps
        self.t_min = t_min
        self.pos_init_mode = pos_init_mode
        self.destination_prediction = destination_prediction
        self.sampling_strategy = sampling_strategy

    def interdependency_modeling(
        self,
        # time,
        time1,
        time2,
        protein_pos,  # transform from the orginal BFN codebase
        protein_v,  # transform from
        batch_protein,  # index for protein
        theta_h_t,
        mu_pos_t,
        batch_ligand,  # index for ligand
        theta_bond_t,
        ligand_bond_index,
        batch_ligand_bond,
        return_all=False,  # legacy from targetdiff
        fix_x=False,
        gen_flag_lig=None,
    ):
        """
        Args:
            time: [node_num x batch_size, 1] := [N_ligand, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [node_num x batch_size, protein_atom_feature_dim] := [N_protein, 27]
            batch_protein: [node_num x batch_size] := [N_protein]
            theta_h_t: [node_num x batch_size, atom_type] := [N_ligand, 13]
            mu_pos_t: [node_num x batch_size, 3] := [N_ligand, 3]
            batch_ligand: [node_num x batch_size] := [N_ligand]
            gamma_coord: [node_num x batch_size, 1] := [N_ligand, 1]
        """
        K = self.num_classes  # ligand_atom_feature_dim
        theta_h_t = 2 * theta_h_t - 1  # from 1/K \in [0,1] to 2/K-1 \in [-1,1]
        init_ligand_v = theta_h_t

        if theta_bond_t is not None:
            KE = self.num_bond_classes
            theta_bond_t = 2 * theta_bond_t - 1
            # ligand_bond_matrix = get_batch_type_pmf_matrix(batch_ligand, ligand_bond_index, theta_bond_t, batch_ligand_bond, padding=False)  # Not streamlined
            # init_ligand_bond = torch.cat([matrix.view(len(matrix), -1) for matrix in ligand_bond_matrix], dim=0).to(theta_bond_t.device)
            # init_ligand_v = torch.cat([init_ligand_v, init_ligand_bond], -1)

        # ---------for targetdiff-----------
        batch_size = batch_ligand.max().item() + 1
        

        # time embedding [simple, sin, rbf, learn]
        if self.time_emb_dim > 0:
            # time_emb = self.time_emb_layer(time)
            # input_ligand_feat = torch.cat([init_ligand_v, time_emb], -1)

            time_emb1 = self.time_emb_layer1(time1)
            time_emb2 = self.time_emb_layer2(time2)
            input_ligand_feat = torch.cat([init_ligand_v, time_emb1, time_emb2], -1)
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)  # [N_protein, self.hidden_dim - 1]
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)  # [N_ligand, self.hidden_dim - 1]
        
        if theta_bond_t is not None:
            h_bond = self.ligand_bond_emb(theta_bond_t)

        if self.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim ]
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim]


        if protein_pos is not None:
            h_all, pos_all, batch_all, mask_ligand, mask_gen, protein_index_in_ctx, ligand_index_in_ctx = compose_context(
                h_protein=h_protein,
                h_ligand=init_ligand_h,
                pos_protein=protein_pos,
                pos_ligand=mu_pos_t,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                gen_flag_lig=gen_flag_lig,
            )
            bond_index_in_all = ligand_index_in_ctx[ligand_bond_index]
        else:
            h_all, pos_all, batch_all = init_ligand_h, mu_pos_t, batch_ligand
            mask_ligand = torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool()
            mask_gen = torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool()
            bond_index_in_all = ligand_bond_index


        if theta_bond_t is not None:
            include_protein = protein_pos is not None
            # node_time = time[batch_all].squeeze(-1)
            # bond_time = time[batch_ligand_bond].squeeze(-1)

            node_time1 = time1[batch_all].squeeze(-1)
            node_time2 = time2[batch_all].squeeze(-1)
            bond_time1 = time1[batch_ligand_bond].squeeze(-1)
            bond_time2 = time2[batch_ligand_bond].squeeze(-1)
            outputs = self.unio2net(
                h=h_all, x=pos_all,
                bond_index=bond_index_in_all, h_bond=h_bond,
                mask_ligand=mask_ligand,
                mask_gen=mask_gen,
                batch=batch_all,
                # node_time=node_time,
                # bond_time=bond_time,
                node_time1=node_time1,
                node_time2=node_time2,
                bond_time1=bond_time1,
                bond_time2=bond_time2,
                include_protein=include_protein,
                return_all=return_all
            )
        else:
            outputs = self.unio2net(
                h=h_all, x=pos_all,
                mask_ligand=mask_ligand,
                mask_gen=mask_gen,
                batch=batch_all, 
                return_all=return_all, 
                fix_x=fix_x)


        final_pos, final_h = (
            outputs["x"],
            outputs["h"],
        )  # shape of the pos and shape of h
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]

        if not self.destination_prediction:  # True for self.destination_prediction
            raise ValueError(f'not implement for no destination_prediction!')
        else:
            coord_pred = final_ligand_pos #add destination prediction. 

        final_ligand_v = self.v_inference(final_ligand_h)  # [N_ligand, 13]
        p0_h = torch.nn.functional.softmax(final_ligand_v, dim=-1)  # [N_ligand, 13]

        if theta_bond_t is not None:
            final_ligand_bond = outputs['h_bond']

            # bond_mask = mask_gen[bond_index_in_all[0]] & mask_gen[bond_index_in_all[1]] 
            # final_gen_bond = final_ligand_bond[bond_mask]

            final_ligand_e = self.bond_inference(final_ligand_bond)
            p0_e = torch.nn.functional.softmax(final_ligand_e, dim=-1)
        else:
            p0_e = None


        return coord_pred, p0_h, p0_e

    def reconstruction_loss_one_step(
        self,
        # t,  # [N_ligand, 1]
        t1,
        t2,
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
        ligand_bond_type,
        ligand_bond_index,
        batch_ligand_bond,
    ):
        # TODO: implement reconstruction loss (but do we really need it?)
        return self.loss_one_step(
            # t=t,
            t1=t1,
            t2=t2,
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos,
            ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            ligand_bond_type=ligand_bond_type,
            ligand_bond_index=ligand_bond_index,
            batch_ligand_bond=batch_ligand_bond,
        )

    def loss_one_step(
        self,
        # t,  # [N_ligand, 1]
        t1,
        t2,
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
        ligand_bond_type,
        ligand_bond_index,
        batch_ligand_bond,
    ):
        K = self.num_classes
        assert ligand_v.max().item() < K, f"Error: {ligand_v.max().item()} >= {K}"
        ligand_v = F.one_hot(ligand_v, K).float()  # [N, K]

        if ligand_bond_type is not None:
            KE = self.num_bond_classes
            ligand_connectivity = (ligand_bond_type != 0).long()
            assert ligand_bond_type.max().item() < KE, f"Error: {ligand_bond_type.max().item()} >= {KE}"
            ligand_bond_type = F.one_hot(ligand_bond_type, KE).float()  # [Nb, KE]


        mu_coord = self.continuous_var_interpolation_update(
            t1, x=ligand_pos, s0=self.c_s0, s1=self.c_s1,
        )  # [N, 3], [N, 1], gamma_coord is not used
        theta = self.discrete_var_interpolation_update(
            t1, x=ligand_v, K=K, s1=self.d_s1
        )  # [N, K]
        if ligand_bond_type is not None:
            t_graph = scatter_mean(t1, batch_ligand, dim=0)
            t_bond = t_graph.index_select(0, batch_ligand_bond)
            theta_bond = self.discrete_var_interpolation_update(
                t_bond, x=ligand_bond_type, K=KE, s1=self.d_s1
            )  # [Nb, KE]
        else:
            theta_bond = None



        # TODO:  modify for bonds
        gen_flag_lig = torch.ones([batch_ligand.size(0)], device=batch_ligand.device, dtype=torch.float32) # float, 1.0 for generate
        use_random_mask = self.use_random_mask
        pm = self.pm
        pam = self.pam
        if use_random_mask:
            if torch.rand(1) >= pm:
                pass
            else:
                # mask = torch.rand_like(gen_flag_lig) < pam  # random mask
                # gen_flag_lig[mask] = 0.0

                substructure_size = torch.randint(1, 20, size=()).to(ligand_pos.device)  # BFS mask
                mask = bfs_substructure_mask(ligand_pos, batch_ligand, substructure_size=substructure_size, distance_threshold=1.8)
                gen_flag_lig[~mask] = 0.0
        
        mu_coord = mu_coord * gen_flag_lig[...,None] + ligand_pos * (1.0 - gen_flag_lig[...,None])
        theta = theta * gen_flag_lig[...,None] + ligand_v * (1.0 - gen_flag_lig[...,None])

        if theta_bond is not None:
            gen_flag_bond = ((gen_flag_lig[ligand_bond_index[0]] > 0) & (gen_flag_lig[ligand_bond_index[1]] > 0)).float()

            theta_bond = theta_bond * gen_flag_bond[..., None] + ligand_bond_type * (1.0 - gen_flag_bond[..., None])



        coord_pred, p0_h, p0_e = self.interdependency_modeling(
            # time=t,
            time1 = t1,
            time2 = t2,
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            theta_h_t=theta,
            mu_pos_t=mu_coord,
            batch_ligand=batch_ligand,
            theta_bond_t = theta_bond,
            ligand_bond_index = ligand_bond_index,
            batch_ligand_bond = batch_ligand_bond,
            gen_flag_lig=gen_flag_lig,
        )  # [N, 3], [N, K], [?]


        # 3. Compute reweighted loss (previous [N,] now [B,])
        if not self.use_discrete_t:  # True for self.use_discrete_t
            raise NotImplementedError
        else:
            # delta_t = 1 / self.discrete_steps
            # t_next = torch.clamp(t + delta_t, max=1.0)
            
            t_next = t1  # loss1
            # t_next = (t1+t2)/2  # loss2
            # t_next = t2  # loss10

            closs = self.dtime4continuous_interpolation_loss(
                t=t_next,
                N=self.discrete_steps,
                x_pred=coord_pred,
                x=ligand_pos,
                s0=self.c_s0,
                s1=self.c_s1,
                segment_ids=batch_ligand,
            )

            dloss = self.dtime4discrete_interpolation_loss_prob(
                t=t_next,
                N=self.discrete_steps,
                p_0=p0_h,
                one_hot_x=ligand_v,
                K=K,
                s1=self.d_s1,
                segment_ids=batch_ligand,
            )

            if ligand_bond_type is not None:
                # t_graph = scatter_mean(t1, batch_ligand, dim=0)
                # t_bond = t_graph.index_select(0, batch_ligand_bond)
                t_next_graph = scatter_mean(t_next, batch_ligand, dim=0)
                t_next_bond = t_next_graph.index_select(0, batch_ligand_bond)
                eloss = self.dtime4discrete_interpolation_loss_prob(
                    t=t_next_bond,
                    N=self.discrete_steps,
                    p_0=p0_e,
                    one_hot_x=ligand_bond_type,
                    K=KE,
                    s1=self.d_s1,
                    segment_ids=batch_ligand_bond,
                )
            else:
                eloss = torch.zeros_like(closs)

        return closs.mean(), dloss.mean(), eloss.mean()


    def sample(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        batch_ligand,
        n_nodes,  # B
        ligand_bond_index=None,
        batch_ligand_bond=None,
        sample_steps=1000,
        desc='',
        ligand_pos_ref=None,
        ligand_v_ref=None,
        ligand_bond_index_ref=None,
        ligand_bond_type_ref=None,
        ligand_bond_batch_ref=None,
        ligand_batch_ref=None,
        gen_flag_lig=None,
    ):
        """
        The function implements the sampling procedure
        Args:
            t: should be a scalar tensor or the shape of [node_num x batch_size, 1] := [N, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [N_protein, protein_atom_feature_dim] := [N_protein, 27]
            batch_ligand / protein: segment_ids for ligand / protein
        """

        K = self.num_classes


        ### prior for x ###
        mu_pos_t = self.c_s0 * torch.randn((len(batch_ligand), 3)).to(self.device)

        ### prior for h ###
        # a_dirichlet = torch.ones((n_nodes, K)) / K
        a_dirichlet = torch.ones((len(batch_ligand), K))
        dirichlet_dist = torch.distributions.Dirichlet(a_dirichlet)
        theta_h_t = dirichlet_dist.sample().to(self.device)


        if ligand_bond_index is not None:
            KE = self.num_bond_classes
            # a_dirichlet = torch.ones((n_nodes, KE)) / KE
            a_dirichlet = torch.ones((len(batch_ligand_bond), KE))
            dirichlet_dist = torch.distributions.Dirichlet(a_dirichlet)
            theta_bond_t = dirichlet_dist.sample().to(self.device)
        else:
            theta_bond_t = None


        theta_traj = []

        if gen_flag_lig is not None:  # fix some parts of the mol
            ligand_v_onehot_ref = F.one_hot(ligand_v_ref, K).float()
            
            # mu_pos_t += torch.mean(ligand_pos * gen_flag_lig[...,None], dim=0)
            # mu_pos_t = mu_pos_t * gen_flag_lig[...,None] + ligand_pos * (1.0 - gen_flag_lig[...,None])
            # theta_h_t = theta_h_t * gen_flag_lig[...,None] + ligand_v_onehot * (1.0 - gen_flag_lig[...,None])

            mu_pos_t = ligand_pos_ref
            theta_h_t = ligand_v_onehot_ref
            batch_ligand = ligand_batch_ref

            if ligand_bond_index is not None:
                ligand_bond_type_onehot_ref = F.one_hot(ligand_bond_type_ref, KE).float()
                theta_bond_t = ligand_bond_type_onehot_ref
                ligand_bond_index = ligand_bond_index_ref
                batch_ligand_bond = ligand_bond_batch_ref



        t_list = torch.tensor([1.0 for _ in range(sample_steps)])
        t_list = torch.softmax(t_list, dim=0)
        t_list = torch.cumsum(t_list, dim=0)
        t_list = t_list / t_list[-1]
        t_list = torch.cat([torch.tensor([0.0]), t_list]).to(self.device)



        for i in trange(1, sample_steps + 1, desc=f'{desc}'):

            # t = torch.ones((n_nodes, 1)).to(self.device) * t_list[i-1]
            t1 = torch.ones((n_nodes, 1)).to(self.device) * t_list[i-1]
            t2 = torch.ones((n_nodes, 1)).to(self.device) * t_list[i]

            if not self.use_discrete_t and not self.destination_prediction:
                # t = torch.clamp(t, min=self.t_min)
                t1 = torch.clamp(t1, min=self.t_min)
                t2 = torch.clamp(t2, min=self.t_min)

            # t = t[batch_ligand]
            t1 = t1[batch_ligand]
            t2 = t2[batch_ligand]

            coord_pred, p0_h_pred, p0_e_pred = self.interdependency_modeling(
                # time=t,
                time1=t1,
                # time2=t2,
                time2=t1,  # t2 is set as t1 during training
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                theta_h_t=theta_h_t,
                mu_pos_t=mu_pos_t, 
                theta_bond_t = theta_bond_t,
                ligand_bond_index = ligand_bond_index,
                batch_ligand_bond = batch_ligand_bond,
                gen_flag_lig=gen_flag_lig,
            )

            theta_traj.append((mu_pos_t, theta_h_t, theta_bond_t))


            if self.sampling_strategy == "end_back_pmf":  # self.sampling_strategy is end_back_pmf
                t = t2
                mu_pos_t = self.continuous_var_interpolation_update(t, x=coord_pred, s0=self.c_s0, s1=self.c_s1)
                theta_h_t = self.discrete_var_interpolation_update(t, x=p0_h_pred, K=K, s1=self.d_s1)
                if ligand_bond_index is not None:
                    t_graph = scatter_mean(t, batch_ligand, dim=0)
                    t_bond = t_graph.index_select(0, batch_ligand_bond)
                    theta_bond_t = self.discrete_var_interpolation_update(t_bond, x=p0_e_pred, K=KE, s1=self.d_s1)
                else:
                    theta_bond_t = None

            else:
                raise NotImplementedError(f"sampling strategy {self.sampling_strategy} not implemented")
            

            if gen_flag_lig is not None:
                mu_pos_t = mu_pos_t * gen_flag_lig[...,None] + ligand_pos_ref * (1.0 - gen_flag_lig[...,None])

                theta_h_t = theta_h_t * gen_flag_lig[...,None] + ligand_v_onehot_ref * (1.0 - gen_flag_lig[...,None])

                if ligand_bond_index is not None:
                    gen_flag_bond = ((gen_flag_lig[ligand_bond_index[0]] > 0) & (gen_flag_lig[ligand_bond_index[1]] > 0)).float()

                    theta_bond_t = theta_bond_t * gen_flag_bond[..., None] + ligand_bond_type_onehot_ref * (1.0 - gen_flag_bond[..., None])


        if gen_flag_lig is not None:
            coord_pred = coord_pred * gen_flag_lig[...,None] + ligand_pos_ref * (1.0 - gen_flag_lig[...,None])

            p0_h_pred = p0_h_pred * gen_flag_lig[...,None] + ligand_v_onehot_ref * (1.0 - gen_flag_lig[...,None])
            if ligand_bond_index is not None:
                gen_flag_bond = ((gen_flag_lig[ligand_bond_index[0]] > 0) & (gen_flag_lig[ligand_bond_index[1]] > 0)).float()
                p0_e_pred = p0_e_pred * gen_flag_bond[..., None] + ligand_bond_type_onehot_ref * (1.0 - gen_flag_bond[..., None])
        theta_traj.append((coord_pred, p0_h_pred, p0_e_pred))

        return theta_traj
