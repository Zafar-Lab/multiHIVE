from scvi.train import AdversarialTrainingPlan
from scvi.module import Classifier
from typing import Callable, Dict, Iterable, Literal, Optional, Union
from scvi.autotune._types import Tunable
from scvi.module.base import BaseModuleClass
from scvi import REGISTRY_KEYS
import torch
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]
from scvi.nn import one_hot
from torch.optim.lr_scheduler import ReduceLROnPlateau

class AdversarialModifiedPlan(AdversarialTrainingPlan):
    def __init__(
            self,
            module: BaseModuleClass,
            *,
            optimizer: Tunable[Literal["Adam", "AdamW", "Custom"]] = "Adam",
            optimizer_creator: Optional[TorchOptimizerCreator] = None,
            lr: Tunable[float] = 1e-3,
            weight_decay: Tunable[float] = 1e-6,
            n_steps_kl_warmup: Tunable[int] = None,
            n_epochs_kl_warmup: Tunable[int] = 400,
            reduce_lr_on_plateau: Tunable[bool] = False,
            lr_factor: Tunable[float] = 0.6,
            lr_patience: Tunable[int] = 30,
            lr_threshold: Tunable[float] = 0.0,
            lr_scheduler_metric: Literal[
                "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
            ] = "elbo_validation",
            lr_min: float = 0,
            adversarial_classifier: Union[bool, Classifier] = False,
            scale_adversarial_loss: Union[float, Literal["auto"]] = "auto",
            **loss_kwargs,
    ):
        super().__init__(
            module=module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            adversarial_classifier=adversarial_classifier,
            scale_adversarial_loss=scale_adversarial_loss,
            **loss_kwargs,
        )

        if adversarial_classifier is True:
            self.adversarial_classifier1 = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_output_classifier,
                n_layers=2,
                logits=True,
            )
            self.adversarial_classifier2 = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_output_classifier,
                n_layers=2,
                logits=True,
            )
        self.automatic_optimization = False

    def loss_adversarial_classifier(self, z, zr, zp, batch_index, predict_true_class=True):
        """Loss for adversarial classifier."""
        n_classes = self.n_output_classifier
        cls_logits = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z))
        cls_logits1 = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier1(zr))
        cls_logits2 = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier2(zp))

        if predict_true_class:
            cls_target = one_hot(batch_index, n_classes)
        else:
            one_hot_batch = one_hot(batch_index, n_classes)
            # place zeroes where true label is
            cls_target = (~one_hot_batch.bool()).float()
            cls_target = cls_target / (n_classes - 1)

        l_soft = cls_logits * cls_target
        l_soft1 = cls_logits1 * cls_target
        l_soft2 = cls_logits2 * cls_target

        loss = -l_soft.sum(dim=1).mean()-l_soft1.sum(dim=1).mean()-l_soft2.sum(dim=1).mean()

        return loss

    def training_step(self, batch, batch_idx):
        """Training step for adversarial training."""
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
        batch_tensor = batch[REGISTRY_KEYS.BATCH_KEY]

        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
        else:
            opt1, opt2 = opts

        inference_outputs, _, scvi_loss = self.forward(
            batch, loss_kwargs=self.loss_kwargs
        )
        z = inference_outputs["z1"]
        z1r = inference_outputs["z1r"]
        z1p = inference_outputs["z1p"]
        loss = scvi_loss.loss
        # fool classifier if doing adversarial training
        if kappa > 0 and self.adversarial_classifier is not False:
            fool_loss = self.loss_adversarial_classifier(z, z1r, z1p, batch_tensor, False)
            loss += fool_loss * kappa

        self.log("train_loss", loss, on_epoch=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if opt2 is not None:
            loss = self.loss_adversarial_classifier(z.detach(), z1r.detach(), z1p.detach(), batch_tensor, True)
            loss *= kappa
            opt2.zero_grad()
            self.manual_backward(loss)
            opt2.step()
        return loss

    def on_train_epoch_end(self):
        """Update the learning rate via scheduler steps."""
        if "validation" in self.lr_scheduler_metric or not self.reduce_lr_on_plateau:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def on_validation_epoch_end(self) -> None:
        """Update the learning rate via scheduler steps."""
        if (
            not self.reduce_lr_on_plateau
            or "validation" not in self.lr_scheduler_metric
        ):
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def configure_optimizers(self):
        """Configure optimizers for adversarial training."""
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            # config1.update(
            #     {
            #         "lr_scheduler": scheduler1,
            #         "monitor": self.lr_scheduler_metric,
            #     },
            # )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler1,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )

        if self.adversarial_classifier is not False:
            params2 = filter(
                lambda p: p.requires_grad, self.adversarial_classifier.parameters()
            )
            optimizer2 = torch.optim.Adam(
                params2, lr=1e-3, eps=0.01, weight_decay=self.weight_decay
            )
            config2 = {"optimizer": optimizer2}

            # bug in pytorch lightning requires this way to return
            opts = [config1.pop("optimizer"), config2["optimizer"]]
            if "lr_scheduler" in config1:
                # config1["scheduler"] = config1.pop("lr_scheduler")
                # scheds = [config1]
                scheds = [config1["lr_scheduler"]]
                return opts, scheds
            else:
                return opts

        return config1