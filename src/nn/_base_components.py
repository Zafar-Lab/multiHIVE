import torch
from scvi.nn import FCLayers
from torch import nn

from typing import Iterable, Literal
from torch.distributions import Normal

import torch.nn.functional as F
from scvi.nn import one_hot


class Encoder(nn.Module):
    """A helper class to build blocks of fully-connected, normalization and dropout layers."""

    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_input: int,
        n_latent: 20,
        n_cat_list: Iterable[int] = None,
        n_hidden: int = 256,
        dropout_rate: float = 0.1,
        distribution: str = "ln",
        kl_dot_product: bool = False,
        deep_network: bool = False
    ):
        super().__init__()
        self.cat = n_cat_list[0]
        self.kl_dot_product = kl_dot_product
        self.deep_network = deep_network
        n_shared_latent2 = n_latent
        n_shared_latent = n_latent
        n_output = n_latent
        n_hidden_protein = 128

        self.encoder_r_1 = nn.Sequential(
            nn.Linear(n_input + n_cat_list[0], n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
        )

        self.encoder_r_1_deep = nn.Sequential(
            nn.Linear(n_input + n_cat_list[0], n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False)
        )
        self.z_mean_encoder_delta_1 = nn.Linear(n_hidden, n_shared_latent)
        self.z_var_encoder_delta_1 = nn.Linear(n_hidden, n_shared_latent)

        self.encoder_r_2 = nn.Sequential(
            # nn.Linear(n_hidden + n_cat_list[0], n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
        )

        self.encoder_r_2_deep = nn.Sequential(
            # nn.Linear(n_hidden + n_cat_list[0], n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
        )

        self.mean_encoder_delta_2 = nn.Linear(n_hidden, n_shared_latent2)
        self.logvar_encoder_delta_2 = nn.Linear(n_hidden, n_shared_latent2)

        self.gene_encoder = nn.Sequential(
            nn.Linear(n_input_genes + n_cat_list[0], n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False)
        )

        self.gene_encoder_deep = nn.Sequential(
            nn.Linear(n_input_genes + n_cat_list[0], n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
        )

        self.zr_mean_encoder = nn.Linear(n_hidden, n_output)
        self.zr_var_encoder = nn.Linear(n_hidden, n_output)

        self.protein_encoder = nn.Sequential(
            nn.Linear(n_input_proteins + n_cat_list[0], n_hidden_protein),
            nn.BatchNorm1d(n_hidden_protein, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden_protein, n_hidden_protein),
            nn.BatchNorm1d(n_hidden_protein, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
        )

        self.protein_encoder_deep = nn.Sequential(
            nn.Linear(n_input_proteins + n_cat_list[0], n_hidden_protein),
            nn.BatchNorm1d(n_hidden_protein, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden_protein, n_hidden_protein),
            nn.BatchNorm1d(n_hidden_protein, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden_protein, n_hidden_protein),
            nn.BatchNorm1d(n_hidden_protein, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
        )

        self.zp_mean_encoder = nn.Linear(n_hidden_protein, n_output)
        self.zp_var_encoder = nn.Linear(n_hidden_protein, n_output)

        self.encoder_z_1 = nn.Sequential(
            nn.Linear(n_shared_latent2, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
        )

        self.encoder_z_1_deep = nn.Sequential(
            nn.Linear(n_shared_latent2, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
        )
        self.mean_encoder_1 = nn.Linear(n_hidden, n_shared_latent)
        self.logvar_encoder_1 = nn.Linear(n_hidden, n_shared_latent)

        self.distribution = distribution
        self.z_transformation = nn.Softmax(dim=-1)
        self.zr_transformation = nn.Softmax(dim=-1)
        self.zp_transformation = nn.Softmax(dim=-1)

    def reparameterize_transformation(self, mu, var):
        """Reparameterization trick to sample from a normal distribution."""
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(self, gene: torch.Tensor, protein: torch.Tensor, data: torch.Tensor,  *cat_list: int):
        batch_onehot_data = one_hot(*cat_list, self.cat)
        data1 = torch.cat((data, batch_onehot_data), dim=-1)

        if self.deep_network:
            r_1 = self.encoder_r_1_deep(data1)
        else:
            r_1 = self.encoder_r_1(data1)

        delta_mu_1 = self.z_mean_encoder_delta_1(r_1)
        delta_logvar_1 = self.z_var_encoder_delta_1(r_1)
        delta_logvar_1 = F.hardtanh(delta_logvar_1, -7., 2.)

        if self.deep_network:
            r_2 = self.encoder_r_2_deep(r_1)
        else:
            r_2 = self.encoder_r_2(r_1)

        delta_mu_2 = self.mean_encoder_delta_2(r_2)
        delta_logvar_2 = self.logvar_encoder_delta_2(r_2)
        delta_logvar_2 = F.hardtanh(delta_logvar_2, -7., 2.)
        delta_var_2 = torch.exp(0.5 * delta_logvar_2) + 1e-4

        q_z2 = Normal(delta_mu_2, delta_var_2.sqrt())
        untran_z2 = q_z2.rsample()
        z2 = untran_z2

        if self.deep_network:
            h_1 = self.encoder_z_1_deep(z2)
        else:
            h_1 = self.encoder_z_1(z2)

        mu_1 = self.mean_encoder_1(h_1)
        logvar_1 = self.logvar_encoder_1(h_1)
        z1_mu = delta_mu_1 + mu_1
        z1_var = torch.exp(0.5 * (delta_logvar_1 + logvar_1)) + 1e-4
        q_z1 = Normal(z1_mu, z1_var.sqrt())
        untran_z1 = q_z1.rsample()
        z1 = untran_z1
        KL_z_2 = 0.5 * (delta_mu_2 ** 2 + torch.exp(delta_logvar_2) - delta_logvar_2 - 1).sum(-1)
        KL_z_1 = 0.5 * (delta_mu_1 ** 2 / torch.exp(logvar_1) + torch.exp(delta_logvar_1) - delta_logvar_1 - 1).sum(
            -1)

        if self.deep_network:
            r_1_gene = self.gene_encoder_deep(torch.cat((gene, batch_onehot_data), dim=-1))
        else:
            r_1_gene = self.gene_encoder(torch.cat((gene, batch_onehot_data), dim=-1))
        delta_mu_1_gene = self.zr_mean_encoder(r_1_gene)
        delta_logvar_1_gene = self.zr_var_encoder(r_1_gene)
        delta_logvar_1_gene = F.hardtanh(delta_logvar_1_gene, -7., 2.)
        delta_var_1_gene = torch.exp(0.5 * delta_logvar_1_gene) + 1e-4
        q_z1r = Normal(delta_mu_1_gene, delta_var_1_gene.sqrt())
        untran_z1r = q_z1r.rsample()
        z1r = untran_z1r
        KL_z_1r = 0.5 * (delta_mu_1_gene ** 2 + torch.exp(delta_logvar_1_gene) - delta_logvar_1_gene - 1).sum(-1)

        if self.deep_network:
            r_1_protein = self.protein_encoder_deep(torch.cat((protein, batch_onehot_data), dim=-1))
        else:
            r_1_protein = self.protein_encoder(torch.cat((protein, batch_onehot_data), dim=-1))
        delta_mu_1_protein = self.zp_mean_encoder(r_1_protein)
        delta_logvar_1_protein = self.zp_var_encoder(r_1_protein)
        delta_logvar_1_protein = F.hardtanh(delta_logvar_1_protein, -7., 2.)
        delta_var_1_protein = torch.exp(0.5 * delta_logvar_1_protein) + 1e-4
        q_z1p = Normal(delta_mu_1_protein, delta_var_1_protein.sqrt())
        untran_z1p = q_z1p.rsample()
        z1p = untran_z1p

        KL_z_1p = 0.5 * (
            delta_mu_1_protein ** 2 + torch.exp(delta_logvar_1_protein) - delta_logvar_1_protein - 1).sum(-1)

        KL = KL_z_1 + KL_z_2 + KL_z_1r + KL_z_1p

        if(self.kl_dot_product):
            KL = KL + + 0.6 * torch.abs((z1 * z1p).sum(dim=1)) + 0.6 * torch.abs((z1 * z1r).sum(dim=1))


        latent = {}
        untran_latent = {}

        latent["z1"] = z1
        latent["z2"] = z2
        latent["kl"] = KL
        latent["z1r"] = z1r
        latent["z1p"] = z1p

        untran_latent["z1"] = untran_z1
        untran_latent["z2"] = untran_z2
        untran_latent["z1r"] = untran_z1r
        untran_latent["z1p"] = untran_z1p

        return q_z1, q_z2, latent, untran_latent, q_z1r, q_z1p


class Decoder(nn.Module):
    """A helper class to build custom decoders depending on which loss was passed."""

    def __init__(
        self,
        n_input: int,
        n_output_genes: int,
        n_output_proteins: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        use_batch_norm: float = True,
        use_layer_norm: float = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
    ):
        super().__init__()

        self.n_output_genes = n_output_genes
        self.n_output_proteins = n_output_proteins

        linear_args = {
            "n_layers": 1,
            "use_activation": False,
            "use_batch_norm": False,
            "use_layer_norm": False,
            "dropout_rate": 0,
        }

        n_shared_latent = 20
        n_input = 20

        self.px_decoder = FCLayers(
            n_in=n_input + n_shared_latent,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean gamma
        self.px_scale_decoder = FCLayers(
            n_in=n_hidden + n_input + n_shared_latent,
            n_out=n_output_genes,
            n_cat_list=n_cat_list,
            **linear_args,
        )
        if scale_activation == "softmax":
            self.px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            self.px_scale_activation = nn.Softplus()

        n_hidden_protein = 256
        # background mean first decoder
        self.py_back_decoder = FCLayers(
            n_in=n_input + n_shared_latent,
            n_out=n_hidden_protein,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden_protein,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        # background mean parameters second decoder
        self.py_back_mean_log_alpha = FCLayers(
            n_in=n_hidden_protein + n_input + n_shared_latent,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )
        self.py_back_mean_log_beta = FCLayers(
            n_in=n_hidden_protein + n_input + n_shared_latent,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        # foreground increment decoder step 1
        self.py_fore_decoder = FCLayers(
            n_in=n_input + n_shared_latent,
            n_out=n_hidden_protein,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden_protein,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        # foreground increment decoder step 2
        self.py_fore_scale_decoder = FCLayers(
            n_in=n_hidden_protein + n_input + n_shared_latent,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
            activation_fn=nn.ReLU,
        )

        # dropout (mixture component for proteins, ZI probability for genes)
        self.sigmoid_decoder = FCLayers(
            n_in=n_input + n_input + n_shared_latent,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.px_dropout_decoder_gene = FCLayers(
            n_in=n_hidden + n_input + n_shared_latent,
            n_out=n_output_genes,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        self.py_background_decoder = FCLayers(
            n_in=n_hidden + n_input + n_shared_latent,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

    def forward(self, z: torch.Tensor, zr: torch.Tensor, zp: torch.Tensor, library_gene: torch.Tensor, *cat_list: int):
        px_ = {}
        py_ = {}

        px = self.px_decoder(torch.cat([z, zr], dim=-1), *cat_list)
        px_cat_z = torch.cat([px, z, zr], dim=-1)
        unnorm_px_scale = self.px_scale_decoder(px_cat_z, *cat_list)
        px_["scale"] = self.px_scale_activation(unnorm_px_scale)
        px_["rate"] = library_gene * px_["scale"]

        py_back = self.py_back_decoder(torch.cat([z, zp], dim=-1), *cat_list)
        py_back_cat_z = torch.cat([py_back, z, zp], dim=-1)
        py_["back_alpha"] = self.py_back_mean_log_alpha(py_back_cat_z, *cat_list)
        py_["back_beta"] = torch.exp(
            self.py_back_mean_log_beta(py_back_cat_z, *cat_list)
        )
        log_pro_back_mean = Normal(py_["back_alpha"], py_["back_beta"]).rsample()
        py_["rate_back"] = torch.exp(log_pro_back_mean)

        py_fore = self.py_fore_decoder(torch.cat([z, zp], dim=-1), *cat_list)
        py_fore_cat_z = torch.cat([py_fore, z, zp], dim=-1)
        py_["fore_scale"] = (
            self.py_fore_scale_decoder(py_fore_cat_z, *cat_list) + 1 + 1e-8
        )
        py_["rate_fore"] = py_["rate_back"] * py_["fore_scale"]

        p_mixing = self.sigmoid_decoder(torch.cat([z, zr, zp], dim=-1), *cat_list)
        p_mixing_cat_z = torch.cat([p_mixing, z], dim=-1)
        px_["dropout"] = self.px_dropout_decoder_gene(torch.cat([p_mixing_cat_z, zr], dim=-1), *cat_list)
        py_["mixing"] = self.py_background_decoder(torch.cat([p_mixing_cat_z, zp], dim=-1), *cat_list)

        protein_mixing = 1 / (1 + torch.exp(-py_["mixing"]))
        py_["scale"] = torch.nn.functional.normalize(
            (1 - protein_mixing) * py_["rate_fore"], p=1, dim=-1
        )

        return (px_, py_, log_pro_back_mean)



