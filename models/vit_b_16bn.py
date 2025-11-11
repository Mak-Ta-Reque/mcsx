import os
import torch
import torch.nn as nn

from .vit_b_16 import (
	safe_divide,
	forward_hook,
	RelProp,
	mySequential,
	myLinear,
	myLayerNorm,
	myDropout,
	ActivationMode,
	_act_fn,
	PatchEmbed,
	MLP,
	MultiHeadSelfAttention,
	transfer_from_torchvision_vit as _transfer_from_tv,
)
from utils.train_config import get_train_config
__all__ = [
	'ViTB16BN',
	'vit_b_16_bn',
	'transfer_from_torchvision_vit_bn',
	'load_vit_b_16bn_model_normal',
	'load_vit_b_16bn_model_manipulated',
]


class TokenBatchNorm(RelProp):
	def __init__(self, dim, eps=1e-5, momentum=0.1):
		super().__init__()
		self.dim = dim
		self.eps = eps
		self.momentum = momentum
		self.weight = nn.Parameter(torch.ones(1, 1, dim))
		self.bias = nn.Parameter(torch.zeros(1, 1, dim))
		self.register_buffer('running_mean', None)
		self.register_buffer('running_var', None)
		self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

	def _ensure_running_stats(self, x):
		_, num_tokens, _ = x.shape
		device = x.device
		dtype = x.dtype
		if self.running_mean is None or self.running_mean.size(1) != num_tokens:
			self.running_mean = torch.zeros(1, num_tokens, self.dim, device=device, dtype=dtype)
			self.running_var = torch.ones(1, num_tokens, self.dim, device=device, dtype=dtype)
			self.num_batches_tracked.zero_()
		else:
			if self.running_mean.device != device or self.running_mean.dtype != dtype:
				self.running_mean = self.running_mean.to(device=device, dtype=dtype)
				self.running_var = self.running_var.to(device=device, dtype=dtype)

	def forward(self, x):
		self._ensure_running_stats(x)
		if self.training:
			with torch.no_grad():
				self.num_batches_tracked += 1
			batch_mean = x.mean(dim=0, keepdim=True)
			batch_var = x.var(dim=0, unbiased=False, keepdim=True)
			with torch.no_grad():
				if self.momentum is None:
					count = max(int(self.num_batches_tracked.item()), 1)
					momentum = 1.0 / float(count)
				else:
					momentum = self.momentum
		else:
			if self.num_batches_tracked.item() == 0:
				batch_mean = x.mean(dim=0, keepdim=True)
				batch_var = x.var(dim=0, unbiased=False, keepdim=True)
			else:
				batch_mean = self.running_mean
				batch_var = self.running_var
		with torch.no_grad():
			if self.training:
				self.running_mean.lerp_(batch_mean, momentum)
				self.running_var.lerp_(batch_var, momentum)

		normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
		return normalized * self.weight + self.bias

	def relprop(self, R, alpha=1, create_graph=False):
		return R


class TransformerEncoderBlockBN(RelProp):
	def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, activation_wrapper=None):
		super().__init__()
		self.norm1 = myLayerNorm(dim)
		self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
		self.drop_path1 = myDropout(drop)

		self.norm2 = myLayerNorm(dim)
		self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop, activation_wrapper=activation_wrapper)
		self.drop_path2 = myDropout(drop)

		self.block_bn = TokenBatchNorm(dim)

	def forward(self, x):
		x = x + self.drop_path1(self.attn(self.norm1(x)))
		x = x + self.drop_path2(self.mlp(self.norm2(x)))
		x = self.block_bn(x)
		return x

	def relprop(self, R, alpha=1, create_graph=False):
		R = self.block_bn.relprop(R, alpha, create_graph=create_graph)

		R_main = R * 0.5
		R_branch = R - R_main

		R_mlp = self.drop_path2.relprop(R_branch, alpha, create_graph=create_graph)
		R_mlp = self.mlp.relprop(R_mlp, alpha, create_graph=create_graph)
		R_mlp = self.norm2.relprop(R_mlp, alpha, create_graph=create_graph)

		R_attn = self.drop_path1.relprop(R_main, alpha, create_graph=create_graph)
		R_attn = self.attn.relprop(R_attn, alpha, create_graph=create_graph)
		R_attn = self.norm1.relprop(R_attn, alpha, create_graph=create_graph)

		return R_attn + R_mlp


class ViTB16BN(nn.Module):
	def __init__(
		self,
		img_size=224,
		patch_size=16,
		in_chans=3,
		num_classes=10,
		embed_dim=768,
		depth=12,
		num_heads=12,
		mlp_ratio=4.0,
		drop_rate=0.0,
		attn_drop_rate=0.0,
		drop_path_rate=0.0,
	):
		super().__init__()
		self.activationmode = ActivationMode.GELU
		self.activation_wrapper = [_act_fn(self.activationmode)]

		self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
		num_patches = self.patch_embed.num_patches

		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
		self.pos_drop = myDropout(drop_rate)

		blocks = []
		for _ in range(depth):
			blocks.append(
				TransformerEncoderBlockBN(
					dim=embed_dim,
					num_heads=num_heads,
					mlp_ratio=mlp_ratio,
					drop=drop_rate,
					attn_drop=attn_drop_rate,
					activation_wrapper=self.activation_wrapper,
				)
			)
		self.blocks = mySequential(*blocks)

		self.norm = myLayerNorm(embed_dim)
		self.fc = myLinear(embed_dim, num_classes)

		nn.init.trunc_normal_(self.pos_embed, std=0.02)
		nn.init.trunc_normal_(self.cls_token, std=0.02)
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.trunc_normal_(m.weight, std=0.02)
			if m.bias is not None:
				nn.init.zeros_(m.bias)
		elif isinstance(m, nn.LayerNorm):
			nn.init.zeros_(m.bias)
			nn.init.ones_(m.weight)
		elif isinstance(m, TokenBatchNorm):
			nn.init.ones_(m.weight)
			nn.init.zeros_(m.bias)

	def set_softplus(self, beta=1.0):
		self.activationmode = ActivationMode.SOFTPLUS
		self.activation_wrapper[0] = _act_fn(self.activationmode, beta=beta)

	def set_relu(self):
		self.activationmode = ActivationMode.RELU
		self.activation_wrapper[0] = _act_fn(self.activationmode)

	def set_gelu(self):
		self.activationmode = ActivationMode.GELU
		self.activation_wrapper[0] = _act_fn(self.activationmode)

	def forward(self, x):
		B = x.size(0)
		x = self.patch_embed(x)
		cls = self.cls_token.expand(B, -1, -1)
		x = torch.cat((cls, x), dim=1)
		x = x + self.pos_embed
		x = self.pos_drop(x)

		x = self.blocks(x)
		x = self.norm(x)
		cls = x[:, 0]
		out = self.fc(cls)
		return out

	def forward_withoutfcl(self, x):
		B = x.size(0)
		x = self.patch_embed(x)
		cls = self.cls_token.expand(B, -1, -1)
		x = torch.cat((cls, x), dim=1)
		x = x + self.pos_embed
		x = self.pos_drop(x)
		x = self.blocks(x)
		x = self.norm(x)
		cls = x[:, 0]
		return cls

	def forward_feature(self, x):
		B = x.size(0)
		x = self.patch_embed(x)
		cls = self.cls_token.expand(B, -1, -1)
		x = torch.cat((cls, x), dim=1)
		x = x + self.pos_embed
		x = self.pos_drop(x)
		x = self.blocks(x)
		return x

	def relprop(self, R, alpha=1, create_graph=False, break_at_basicblocks=False):
		R = self.fc.relprop(R, alpha, create_graph=create_graph)

		B = R.shape[0]
		num_tokens = self.pos_embed.size(1)
		R_tokens = R.new_zeros((B, num_tokens, R.shape[-1]))
		R_tokens[:, 0, :] = R

		R = self.norm.relprop(R_tokens, alpha, create_graph=create_graph)
		R = self.blocks.relprop(R, alpha, create_graph=create_graph)

		R_cls = R[:, :1, :]
		R_patches = R[:, 1:, :]

		R_patches = self.patch_embed.relprop(R_patches, alpha, create_graph=create_graph)

		return R_patches


def transfer_from_torchvision_vit_bn(local_model: ViTB16BN, tv_model) -> int:
	transfers = _transfer_from_tv(local_model, tv_model)
	return transfers


def vit_b_16_bn(**kwargs):
	return ViTB16BN(**kwargs)


def _env_value(keys, default, cast):
	for key in keys:
		val = os.getenv(key)
		if val is not None:
			try:
				return cast(val)
			except (TypeError, ValueError):
				continue
	return default


def _vit_bn_config_from_env():
	img_size = _env_value(['VIT_BN_IMG_SIZE', 'VIT_IMG_SIZE'], 32, int)
	patch_size = _env_value(['VIT_BN_PATCH_SIZE', 'VIT_PATCH_SIZE'], 4, int)
	embed_dim = _env_value(['VIT_BN_EMBED_DIM', 'VIT_EMBED_DIM'], 768, int)
	depth = _env_value(['VIT_BN_DEPTH', 'VIT_DEPTH'], 12, int)
	num_heads = _env_value(['VIT_BN_NUM_HEADS', 'VIT_NUM_HEADS'], 12, int)
	mlp_ratio = _env_value(['VIT_BN_MLP_RATIO', 'VIT_MLP_RATIO'], 4.0, float)
	drop_rate = _env_value(['VIT_BN_DROP', 'VIT_DROP'], 0.0, float)
	attn_drop_rate = _env_value(['VIT_BN_ATTN_DROP', 'VIT_ATTN_DROP'], 0.0, float)
	drop_path_rate = _env_value(['VIT_BN_DROP_PATH', 'VIT_DROP_PATH'], 0.0, float)
	return {
		'img_size': img_size,
		'patch_size': patch_size,
		'embed_dim': embed_dim,
		'depth': depth,
		'num_heads': num_heads,
		'mlp_ratio': mlp_ratio,
		'drop_rate': drop_rate,
		'attn_drop_rate': attn_drop_rate,
		'drop_path_rate': drop_path_rate,
	}


def _build_vit_bn(device, num_classes, **cfg):
	model = vit_b_16_bn(num_classes=num_classes, **cfg).to(device)
	return model


def _try_load_torchvision_vit(device):
	try:
		import torchvision
		weights_attr = getattr(torchvision.models, 'ViT_B_16_Weights', None)
		if weights_attr is not None:
			weights = weights_attr.DEFAULT
			return torchvision.models.vit_b_16(weights=weights).to(device)
		tv = torch.hub.load('pytorch/vision', 'vit_b_16', pretrained=True)
		return tv.to(device)
	except Exception:
		return None


def _auto_train_vit_bn(path, device, num_classes, dataset_enum, cfg):
	from load import load_data
	from torch.utils.data import DataLoader, TensorDataset

	epochs, lr, batch_size = get_train_config(300, 1e-4, 256, specific_prefix='VIT_BN_AUTO')

	x_test, y_test, x_train, y_train = load_data(dataset_enum, test_only=False, shuffle_test=True)
	train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

	model = _build_vit_bn(device, num_classes, **cfg)

	tv_model = _try_load_torchvision_vit(device)
	if tv_model is not None:
		try:
			transferred = transfer_from_torchvision_vit_bn(model, tv_model)
			if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
				print(f"[vit_b_16_bn init] transferred {transferred} tensors from torchvision ViT")
		except Exception as exc:
			if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
				print(f"[vit_b_16_bn init] transfer failed: {exc}")

	criterion = nn.CrossEntropyLoss()
	head_only = os.getenv('VIT_BN_TRAIN_HEAD_ONLY', os.getenv('VIT_TRAIN_HEAD_ONLY', '0')) == '1'
	if head_only:
		for param in model.parameters():
			param.requires_grad = False
		for param in model.fc.parameters():
			param.requires_grad = True
		head_lr = _env_value(['VIT_BN_HEAD_LR', 'VIT_HEAD_LR'], lr, float)
		optimizer = torch.optim.Adam(model.fc.parameters(), lr=head_lr)
		if os.getenv('VERBOSE_MODEL_LOAD', '0') == '1':
			print(f"[vit_b_16_bn auto-train] head-only training enabled (lr={head_lr})")
	else:
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	for ep in range(epochs):
		model.train()
		loss_sum = 0.0
		steps = 0
		for xb, yb in train_loader:
			xb, yb = xb.to(device), yb.to(device)
			optimizer.zero_grad(set_to_none=True)
			logits = model(xb)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()
			loss_sum += loss.item()
			steps += 1
		if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
			mean_loss = loss_sum / max(steps, 1)
			print(f"[vit_b_16_bn auto-train] epoch {ep + 1}/{epochs} loss={mean_loss:.4f}")

	@torch.no_grad()
	def _accuracy(m):
		m.eval()
		correct = 0
		total = 0
		for xb, yb in test_loader:
			xb, yb = xb.to(device), yb.to(device)
			preds = m(xb).argmax(1)
			correct += (preds == yb).sum().item()
			total += yb.numel()
		return correct / max(total, 1)

	acc = _accuracy(model)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(os.path.join(os.path.dirname(path), 'accuracy.txt'), 'w') as f:
		f.write(f"Accuracy: {acc:.4f}\n")
	torch.save({'state_dict': model.state_dict(), 'meta': {'source': 'vit_b_16_bn_auto', 'num_classes': num_classes, **cfg, 'accuracy': acc}}, path)
	if os.getenv('VERBOSE_MODEL_LOAD', '1') == '1':
		print(f"[vit_b_16_bn auto-train] saved checkpoint to {path} acc={acc:.4f}")


def _load_vit_bn_checkpoint(path, device, *, allow_auto_train, num_classes, dataset_enum, cfg):
	try:
		checkpoint = torch.load(path, map_location=device)
	except (FileNotFoundError, OSError):
		if not allow_auto_train:
			raise
		_auto_train_vit_bn(path, device, num_classes, dataset_enum, cfg)
		checkpoint = torch.load(path, map_location=device)
	return checkpoint


def _finalize_vit_bn_model(device, num_classes, checkpoint, cfg):
	model = _build_vit_bn(device, num_classes, **cfg)
	state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
	missing = model.load_state_dict(state_dict, strict=False)
	if os.getenv('VERBOSE_MODEL_LOAD', '0') == '1':
		print(f"[vit_b_16_bn load] missing keys: {missing.missing_keys}, unexpected: {missing.unexpected_keys}")
	model.eval()
	return model


def load_vit_b_16bn_model_normal(path, device, num_classes=10, dataset_enum=None):
	if dataset_enum is None:
		from utils.config import DatasetEnum
		dataset_enum = DatasetEnum.CIFAR10
	cfg = _vit_bn_config_from_env()
	checkpoint = _load_vit_bn_checkpoint(path, device, allow_auto_train=True, num_classes=num_classes, dataset_enum=dataset_enum, cfg=cfg)
	return _finalize_vit_bn_model(device, num_classes, checkpoint, cfg)


def load_vit_b_16bn_model_manipulated(path, device, num_classes=10, dataset_enum=None):
	if dataset_enum is None:
		from utils.config import DatasetEnum
		dataset_enum = DatasetEnum.CIFAR10
	cfg = _vit_bn_config_from_env()
	checkpoint = _load_vit_bn_checkpoint(path, device, allow_auto_train=False, num_classes=num_classes, dataset_enum=dataset_enum, cfg=cfg)
	return _finalize_vit_bn_model(device, num_classes, checkpoint, cfg)


if __name__ == "__main__":
	model = vit_b_16_bn(num_classes=10, img_size=224)
	x = torch.randn(2, 3, 224, 224)
	logits = model(x)
	print("Logits:", logits.shape)
