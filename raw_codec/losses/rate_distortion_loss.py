from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from raw_codec.data.noise_model import subtract_black_level_from_raw
from raw_codec.data.rawsim_config import resolve_noise_config, resolve_unprocess_config
from raw_codec.losses.loss_zoo import charbonnier_loss, convert_loss_domain, lpips_loss, ms_ssim_loss, ssim_loss
from raw_codec.metrics.bitrate import estimate_bitrate


class RateDistortionLoss(nn.Module):
    """Rate-distortion objective for noisy RAW to linear camera RGB regression."""

    def __init__(
        self,
        loss_cfg: dict | None = None,
        bitrate_cfg: dict | None = None,
        unprocess_cfg: dict | None = None,
        noise_cfg: dict | None = None,
        linear_rgb_has_black_level: bool = False,
        raw_bit_depth: int = 12,
        quant_step_size: float = 1.0,
    ) -> None:
        super().__init__()
        loss_cfg = dict(loss_cfg or {})
        self.lambda_distortion = float(loss_cfg.get("lambda_distortion", loss_cfg.get("lambda_l1", 1.0)))
        self.lambda_mse = float(loss_cfg.get("lambda_mse", 0.0))
        self.lambda_camera_l1 = float(loss_cfg.get("lambda_camera_l1", loss_cfg.get("lambda_linear_l1", 0.0)))
        self.lambda_srgb_l1 = float(loss_cfg.get("lambda_srgb_l1", loss_cfg.get("lambda_display_l1", 0.0)))
        self.lambda_srgb_lpips = float(loss_cfg.get("lambda_srgb_lpips", loss_cfg.get("lambda_display_lpips", 0.0)))
        self.srgb_lpips_start_step = max(int(loss_cfg.get("srgb_lpips_start_step", 0)), 0)
        self.srgb_lpips_ramp_steps = max(int(loss_cfg.get("srgb_lpips_ramp_steps", 0)), 0)
        self.current_step = 0
        self.srgb_lpips_net = str(loss_cfg.get("srgb_lpips_net", loss_cfg.get("lpips_net", "alex"))).lower()
        self.srgb_lpips_resize = int(loss_cfg.get("srgb_lpips_resize", loss_cfg.get("lpips_resize", 0)))
        self.lambda_rate = float(loss_cfg.get("lambda_rate", 0.0))
        self.lambda_rate_initial = float(
            loss_cfg.get("lambda_rate_initial", self.lambda_rate)
        )
        self.lambda_rate_start_step = max(
            int(loss_cfg.get("lambda_rate_start_step", 0)),
            0,
        )
        self.lambda_rate_ramp_steps = max(
            int(loss_cfg.get("lambda_rate_ramp_steps", 0)),
            0,
        )
        self.target_bpp = float(
            loss_cfg.get("target_bpp", loss_cfg.get("rate_budget_bpp", 0.0))
        )
        self.lambda_rate_budget = float(
            loss_cfg.get(
                "lambda_rate_budget",
                loss_cfg.get("rate_budget_weight", 0.0),
            )
        )
        if self.lambda_rate_budget < 0.0:
            raise ValueError("loss.lambda_rate_budget must be non-negative.")
        if self.lambda_rate_budget > 0.0 and self.target_bpp <= 0.0:
            raise ValueError(
                "loss.target_bpp must be positive when lambda_rate_budget is enabled."
            )
        latent_consistency_cfg = dict(loss_cfg.get("latent_consistency", {}) or {})
        self.latent_consistency_enabled = bool(latent_consistency_cfg.get("enabled", False))
        self.lambda_latent_consistency = float(latent_consistency_cfg.get("weight", 0.0))
        self.latent_consistency_loss = str(latent_consistency_cfg.get("loss", "smooth_l1")).lower()
        self.latent_consistency_normalization = str(
            latent_consistency_cfg.get("normalization", "target_rms")
        ).lower()
        self.latent_consistency_scale_floor = float(latent_consistency_cfg.get("scale_floor", 0.1))
        self.latent_consistency_beta = float(latent_consistency_cfg.get("beta", 1.0))
        if self.latent_consistency_scale_floor <= 0.0:
            raise ValueError("loss.latent_consistency.scale_floor must be positive.")
        if self.latent_consistency_beta <= 0.0:
            raise ValueError("loss.latent_consistency.beta must be positive.")
        self.reconstruction_loss = str(loss_cfg.get("reconstruction_loss", "l1")).lower()
        mix_cfg = dict(loss_cfg.get("distortion_mix", loss_cfg.get("l1_mix", {})) or {})
        self.distortion_linear_l1_weight = float(
            loss_cfg.get(
                "distortion_linear_l1_weight",
                loss_cfg.get(
                    "linear_l1_weight",
                    mix_cfg.get("linear_l1_weight", mix_cfg.get("camera_l1_weight", 0.8)),
                ),
            )
        )
        self.distortion_srgb_l1_weight = float(
            loss_cfg.get(
                "distortion_srgb_l1_weight",
                loss_cfg.get("srgb_l1_weight", mix_cfg.get("srgb_l1_weight", 0.2)),
            )
        )
        self.distortion_linear_mse_weight = float(
            loss_cfg.get("distortion_linear_mse_weight", loss_cfg.get("linear_mse_weight", 0.65))
        )
        self.distortion_srgb_mse_weight = float(
            loss_cfg.get("distortion_srgb_mse_weight", loss_cfg.get("srgb_mse_weight", 0.35))
        )
        self.distortion_mse_scale = float(loss_cfg.get("distortion_mse_scale", 255.0**2))
        self.charbonnier_eps = float(loss_cfg.get("charbonnier_eps", 1e-3))
        self.loss_domain = str(loss_cfg.get("domain", loss_cfg.get("loss_domain", "camera"))).lower()
        self.unprocess_cfg = dict(unprocess_cfg or {})
        self.noise_cfg = dict(noise_cfg or {})
        self.linear_rgb_has_black_level = bool(linear_rgb_has_black_level)
        self.bitrate_cfg = dict(bitrate_cfg or {})
        self.raw_bit_depth = int(raw_bit_depth)
        self.quant_step_size = float(quant_step_size)

    @classmethod
    def from_config(cls, config: dict) -> "RateDistortionLoss":
        return cls(
            loss_cfg=config.get("loss", {}),
            bitrate_cfg=config.get("bitrate", {}),
            unprocess_cfg=resolve_unprocess_config(config),
            noise_cfg=resolve_noise_config(config),
            linear_rgb_has_black_level=bool(
                config.get("target", {}).get(
                    "include_black_level",
                    config.get("target", {}).get(
                        "linear_rgb_include_black_level",
                        config.get("target", {}).get("linear_rgb_has_black_level", False),
                    ),
                )
            ),
            raw_bit_depth=int(config.get("model", {}).get("raw_bit_depth", 12)),
            quant_step_size=float(config.get("quantizer", {}).get("step_size", 1.0)),
        )

    def _domain_uses_display_signal(self, domain: str | None = None) -> bool:
        domain = str(domain or self.loss_domain).lower()
        return domain in {"srgb", "s_rgb", "display", "rendered", "rgb"}

    def set_step(self, step: int) -> None:
        self.current_step = max(int(step), 0)

    def current_srgb_lpips_weight(self) -> float:
        if self.lambda_srgb_lpips == 0.0 or self.current_step < self.srgb_lpips_start_step:
            return 0.0
        if self.srgb_lpips_ramp_steps <= 0:
            return self.lambda_srgb_lpips
        progress = min(
            max((self.current_step - self.srgb_lpips_start_step) / float(self.srgb_lpips_ramp_steps), 0.0),
            1.0,
        )
        return self.lambda_srgb_lpips * progress

    def current_rate_weight(self) -> float:
        if self.current_step < self.lambda_rate_start_step:
            return self.lambda_rate_initial
        if self.lambda_rate_ramp_steps <= 0:
            return self.lambda_rate
        progress = min(
            max(
                (self.current_step - self.lambda_rate_start_step)
                / float(self.lambda_rate_ramp_steps),
                0.0,
            ),
            1.0,
        )
        return self.lambda_rate_initial + (
            self.lambda_rate - self.lambda_rate_initial
        ) * progress

    def _remove_black_for_display(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.linear_rgb_has_black_level:
            return tensor
        return subtract_black_level_from_raw(tensor, self.noise_cfg)

    def _prepare_for_domain(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        domain: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._domain_uses_display_signal(domain):
            return self._remove_black_for_display(recon), self._remove_black_for_display(target)
        return recon, target

    def _loss_domain_kwargs(self, batch: dict[str, torch.Tensor] | None = None) -> dict:
        return {
            "domain": self.loss_domain,
            "gamma": float(self.unprocess_cfg.get("gamma", 2.2)),
            "ccm": (
                self.unprocess_cfg.get("ccm")
                if batch is None
                else batch.get("camera_to_srgb", self.unprocess_cfg.get("ccm"))
            ),
            "awb_gains": None if batch is None else batch.get("awb_gains"),
        }

    def _convert_for_reporting(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recon, target = self._prepare_for_domain(recon, target, self.loss_domain)
        return convert_loss_domain(recon, target, **self._loss_domain_kwargs(batch))

    def _uses_linear_srgb_l1_mix(self) -> bool:
        return self.reconstruction_loss in {
            "linear_srgb_l1_mix",
            "camera_srgb_l1_mix",
            "mixed_l1",
            "l1_mix",
        }

    def _uses_elic_mse_mix(self) -> bool:
        return self.reconstruction_loss in {
            "elic_mse_mix",
            "linear_srgb_mse_mix",
            "camera_srgb_mse_mix",
            "mixed_mse",
        }

    def _primary_reconstruction_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self._uses_elic_mse_mix():
            linear_weight = self.distortion_linear_mse_weight
            srgb_weight = self.distortion_srgb_mse_weight
            if linear_weight < 0.0 or srgb_weight < 0.0 or linear_weight + srgb_weight <= 0.0:
                raise ValueError(
                    "distortion linear/sRGB MSE weights must be non-negative and sum to a positive value."
                )
            linear_mse = F.mse_loss(recon, target)
            srgb_mse = self._srgb_mse_loss(recon, target, batch=batch)
            return self.distortion_mse_scale * (
                linear_weight * linear_mse + srgb_weight * srgb_mse
            )

        if self._uses_linear_srgb_l1_mix():
            linear_weight = self.distortion_linear_l1_weight
            srgb_weight = self.distortion_srgb_l1_weight
            if linear_weight < 0.0 or srgb_weight < 0.0 or linear_weight + srgb_weight <= 0.0:
                raise ValueError(
                    "distortion linear/sRGB L1 weights must be non-negative and sum to a positive value."
                )
            camera_l1 = F.l1_loss(recon, target)
            srgb_l1 = self._srgb_l1_loss(recon, target, batch=batch)
            return linear_weight * camera_l1 + srgb_weight * srgb_l1

        domain_kwargs = self._loss_domain_kwargs(batch)
        recon, target = self._prepare_for_domain(recon, target, domain_kwargs["domain"])
        if self.reconstruction_loss in {"charbonnier", "charb", "charbonnier_l1"}:
            return charbonnier_loss(recon, target, eps=self.charbonnier_eps, **domain_kwargs)
        if self.reconstruction_loss in {"ssim", "ssim_loss"}:
            return ssim_loss(recon, target, data_range=1.0, **domain_kwargs)
        if self.reconstruction_loss in {"msssim", "ms_ssim", "msssim_loss", "ms_ssim_loss"}:
            return ms_ssim_loss(recon, target, data_range=1.0, **domain_kwargs)
        recon_loss_domain, target_loss_domain = convert_loss_domain(recon, target, **domain_kwargs)
        return F.l1_loss(recon_loss_domain, target_loss_domain)

    def _srgb_l1_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        recon_srgb, target_srgb = self.to_srgb(recon, target, batch=batch)
        return F.l1_loss(recon_srgb, target_srgb)

    def _srgb_mse_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        recon_srgb, target_srgb = self.to_srgb(recon, target, batch=batch)
        return F.mse_loss(recon_srgb, target_srgb)

    def to_srgb(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render linear camera RGB prediction and target into the configured sRGB domain."""
        recon, target = self._prepare_for_domain(recon, target, "srgb")
        return convert_loss_domain(
            recon,
            target,
            domain="srgb",
            gamma=float(self.unprocess_cfg.get("gamma", 2.2)),
            ccm=(
                self.unprocess_cfg.get("ccm")
                if batch is None
                else batch.get("camera_to_srgb", self.unprocess_cfg.get("ccm"))
            ),
            awb_gains=None if batch is None else batch.get("awb_gains"),
        )

    def _srgb_lpips_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        recon_srgb, target_srgb = self.to_srgb(recon, target, batch=batch)
        if self.srgb_lpips_resize > 0 and recon_srgb.shape[-2:] != (self.srgb_lpips_resize, self.srgb_lpips_resize):
            size = (self.srgb_lpips_resize, self.srgb_lpips_resize)
            recon_srgb = F.interpolate(recon_srgb, size=size, mode="bilinear", align_corners=False)
            target_srgb = F.interpolate(target_srgb, size=size, mode="bilinear", align_corners=False)
        return lpips_loss(
            recon_srgb,
            target_srgb,
            data_range=1.0,
            net=self.srgb_lpips_net,
            domain="camera",
        )

    def _latent_consistency(
        self,
        noisy_latent: torch.Tensor,
        clean_latent: torch.Tensor,
    ) -> torch.Tensor:
        clean_latent = clean_latent.detach().to(device=noisy_latent.device, dtype=noisy_latent.dtype)
        residual = noisy_latent - clean_latent
        if self.latent_consistency_normalization in {"target_rms", "clean_rms", "relative_rms"}:
            scale = clean_latent.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
            scale = scale.clamp_min(self.latent_consistency_scale_floor).to(dtype=residual.dtype)
            residual = residual / scale
        elif self.latent_consistency_normalization not in {"none", "off", "identity"}:
            raise ValueError(
                "Unsupported loss.latent_consistency.normalization="
                f"{self.latent_consistency_normalization!r}."
            )

        zeros = torch.zeros_like(residual)
        if self.latent_consistency_loss in {"smooth_l1", "huber"}:
            return F.smooth_l1_loss(residual, zeros, beta=self.latent_consistency_beta)
        if self.latent_consistency_loss in {"l1", "mae"}:
            return F.l1_loss(residual, zeros)
        if self.latent_consistency_loss in {"l2", "mse"}:
            return F.mse_loss(residual, zeros)
        raise ValueError(f"Unsupported loss.latent_consistency.loss={self.latent_consistency_loss!r}.")

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        target: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        recon = model_output["recon_linear_rgb"]
        quantized_latent = model_output["quantized_latent"]
        qstep = model_output.get("qstep", self.quant_step_size)
        recon_loss_domain, target_loss_domain = self._convert_for_reporting(recon, target, batch)

        l1 = F.l1_loss(recon_loss_domain, target_loss_domain)
        mse = F.mse_loss(recon_loss_domain, target_loss_domain)
        distortion = self._primary_reconstruction_loss(recon, target, batch=batch)

        bitrate = model_output.get("bitrate")
        if bitrate is None:
            bitrate = estimate_bitrate(
                quantized_latent,
                raw_shape=target.shape,
                mode=self.bitrate_cfg.get("mode", "fixed"),
                fixed_bits_per_symbol=float(self.bitrate_cfg.get("fixed_bits_per_symbol", 6.0)),
                raw_bit_depth=self.raw_bit_depth,
                quant_step_size=qstep,
            )

        bpp = bitrate["bpp"].to(device=recon.device, dtype=recon.dtype)
        loss_distortion = self.lambda_distortion * distortion
        loss_mse = self.lambda_mse * mse
        camera_l1 = F.l1_loss(recon, target)
        loss_camera_l1 = self.lambda_camera_l1 * camera_l1
        srgb_l1 = (
            self._srgb_l1_loss(recon, target, batch=batch)
            if self.lambda_srgb_l1 != 0.0
            or self._uses_linear_srgb_l1_mix()
            or self._uses_elic_mse_mix()
            else bpp.new_zeros(())
        )
        srgb_mse = (
            self._srgb_mse_loss(recon, target, batch=batch)
            if self._uses_elic_mse_mix()
            else bpp.new_zeros(())
        )
        loss_srgb_l1 = self.lambda_srgb_l1 * srgb_l1
        srgb_lpips = (
            self._srgb_lpips_loss(recon, target, batch=batch) if self.lambda_srgb_lpips != 0.0 else bpp.new_zeros(())
        )
        srgb_lpips_weight = self.current_srgb_lpips_weight()
        loss_srgb_lpips = srgb_lpips_weight * srgb_lpips
        rate_weight = self.current_rate_weight()
        loss_rate = bpp.new_zeros(()) if rate_weight == 0.0 else rate_weight * bpp
        rate_budget_excess = (
            F.relu(bpp - self.target_bpp)
            if self.lambda_rate_budget > 0.0
            else bpp.new_zeros(())
        )
        loss_rate_budget = self.lambda_rate_budget * rate_budget_excess.square()
        clean_latent = model_output.get("clean_latent_target")
        if self.latent_consistency_enabled and self.lambda_latent_consistency != 0.0:
            if clean_latent is None:
                raise KeyError(
                    "Latent consistency is enabled, but model_output has no 'clean_latent_target'."
                )
            latent_consistency = self._latent_consistency(model_output["latent"], clean_latent)
        else:
            latent_consistency = bpp.new_zeros(())
        loss_latent_consistency = self.lambda_latent_consistency * latent_consistency
        loss = (
            loss_distortion
            + loss_mse
            + loss_camera_l1
            + loss_srgb_l1
            + loss_srgb_lpips
            + loss_rate
            + loss_rate_budget
            + loss_latent_consistency
        )

        items = {
            "loss": loss,
            "loss_distortion": loss_distortion.detach(),
            "loss_mse": loss_mse.detach(),
            "loss_camera_l1": loss_camera_l1.detach(),
            "loss_srgb_l1": loss_srgb_l1.detach(),
            "loss_srgb_lpips": loss_srgb_lpips.detach(),
            "srgb_lpips_weight": bpp.new_tensor(srgb_lpips_weight).detach(),
            "loss_rate": loss_rate.detach(),
            "rate_weight": bpp.new_tensor(rate_weight).detach(),
            "loss_rate_budget": loss_rate_budget.detach(),
            "rate_budget_excess": rate_budget_excess.detach(),
            "target_bpp": bpp.new_tensor(self.target_bpp).detach(),
            "loss_latent_consistency": loss_latent_consistency.detach(),
            "latent_consistency": latent_consistency.detach(),
            "distortion": distortion.detach(),
            "l1": l1.detach(),
            "camera_l1": camera_l1.detach(),
            "srgb_l1": srgb_l1.detach(),
            "srgb_mse": srgb_mse.detach(),
            "srgb_lpips": srgb_lpips.detach(),
            "mse": mse.detach(),
            "bpp": bpp.detach(),
            "compression_ratio": bitrate["compression_ratio"].detach(),
            "total_bits": bitrate["total_bits"].detach(),
            "bits_per_symbol": bitrate["bits_per_symbol"].detach(),
            "sample_bpp": bitrate["sample_bpp"].detach(),
            "estimated_entropy": bitrate["estimated_entropy"].detach(),
            "quantized_latent_unique_count": bitrate["quantized_latent_unique_count"].detach(),
        }
        if "latent_bpp" in bitrate:
            items["latent_bpp"] = bitrate["latent_bpp"].detach()
            items["hyper_bpp"] = bitrate["hyper_bpp"].detach()
        return items
