# System
import os

# Libs
import torch
import torchvision
from torch.utils.data import DataLoader, Subset  # type: ignore

from typing import Optional, Tuple


# Own sources
import utils
from utils.config import DatasetEnum


def load_data(dataset: DatasetEnum, test_only=False, shuffle_test=True):
	"""
	Loads the dataset tensors for the requested dataset enum.

	:param dataset: Dataset identifier from DatasetEnum
	:param test_only: If True, training data is omitted
	:param shuffle_test: If True, test data is shuffled before returning
	:return: x_test, label_test, x_train, label_train
	"""
	device = os.getenv('CUDADEVICE')

	if dataset == DatasetEnum.CIFAR10:
		x_test, label_test, x_train, label_train = load_cifar(test_only=test_only, shuffle_test=shuffle_test)
	elif dataset == DatasetEnum.CIFAR100:
		x_test, label_test, x_train, label_train = load_cifar100(test_only=test_only, shuffle_test=shuffle_test)
	elif dataset == DatasetEnum.GTSRB:
		x_test, label_test, x_train, label_train = load_gtsrb(test_only=test_only, shuffle_test=shuffle_test)
	elif dataset == DatasetEnum.IMAGENET:
		x_test, label_test, x_train, label_train = load_imagenet(test_only=test_only, shuffle_test=shuffle_test)
	else:
		raise Exception(f'Unknown dataset {dataset}')

	x_test, label_test = x_test.to(device), label_test.to(device)
	if not test_only:
		x_train, label_train = x_train.to(device), label_train.to(device)

	return x_test, label_test, x_train, label_train


def shuffle_data(x, label):
	"""
	Shuffle a dataset with a fixed seed to guarantee reproducibility.
	"""
	gen = torch.Generator().manual_seed(200)
	perm = torch.randperm(len(x), generator=gen)
	return x[perm], label[perm]


def split_data(x, label, ratio=0.8):
	"""
	Splits a balanced dataset with a fixed seed and keeps each split balanced.
	"""
	x_split1, label_split1 = [], []
	x_split2, label_split2 = [], []

	gen = torch.Generator().manual_seed(100)
	for cls, count in zip(*label.unique(return_counts=True)):
		cls = int(cls)
		count = int(count)
		idxs = torch.where(label == cls)[0]
		assert len(idxs) == count

		perm = torch.randperm(count, generator=gen)
		idxs = idxs[perm]
		splitpoint = int(count * (1 - ratio))  # Split 1 is test

		x_split1.extend(x[idxs[:splitpoint]])
		label_split1.extend(label[idxs[:splitpoint]])
		x_split2.extend(x[idxs[splitpoint:]])
		label_split2.extend(label[idxs[splitpoint:]])

	x_split1, label_split1 = torch.stack(x_split1), torch.stack(label_split1)
	x_split2, label_split2 = torch.stack(x_split2), torch.stack(label_split2)

	return x_split1, label_split1, x_split2, label_split2


def load_cifar(split_train_test=True, transform=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), test_only=False, shuffle_test=True):
	"""
	Load CIFAR-10 dataset as tensors.
	"""

	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		transform,
	])

	if not test_only:
		cifar_train = torchvision.datasets.CIFAR10(str(utils.config.get_datasetdir(DatasetEnum.CIFAR10)), download=True, train=True, transform=transform)
	cifar_test = torchvision.datasets.CIFAR10(str(utils.config.get_datasetdir(DatasetEnum.CIFAR10)), download=True, train=False, transform=transform)

	if not test_only:
		train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=len(cifar_train), shuffle=True)
	test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=len(cifar_test), shuffle=shuffle_test)

	x_test_, original_label_test_ = next(iter(test_loader))
	if not test_only:
		x_train_, original_label_train_ = next(iter(train_loader))

	if test_only:
		return x_test_, original_label_test_, None, None

	if split_train_test:
		return x_test_, original_label_test_, x_train_, original_label_train_

	x_data = torch.cat((x_test_, x_train_))
	original_label_data = torch.cat((original_label_test_, original_label_train_))
	x_data, label_data = shuffle_data(x_data, original_label_data)
	return x_data, label_data


def load_cifar100(split_train_test=True, transform=torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]), test_only=False, shuffle_test=True):
	"""
	Load CIFAR-100 dataset as tensors.
	"""

	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		transform,
	])

	if not test_only:
		cifar_train = torchvision.datasets.CIFAR100(str(utils.config.get_datasetdir(DatasetEnum.CIFAR100)), download=True, train=True, transform=transform)
	cifar_test = torchvision.datasets.CIFAR100(str(utils.config.get_datasetdir(DatasetEnum.CIFAR100)), download=True, train=False, transform=transform)

	if not test_only:
		train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=len(cifar_train), shuffle=True)
	test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=len(cifar_test), shuffle=shuffle_test)

	x_test_, original_label_test_ = next(iter(test_loader))
	if not test_only:
		x_train_, original_label_train_ = next(iter(train_loader))

	if test_only:
		return x_test_, original_label_test_, None, None

	if split_train_test:
		return x_test_, original_label_test_, x_train_, original_label_train_

	x_data = torch.cat((x_test_, x_train_))
	original_label_data = torch.cat((original_label_test_, original_label_train_))
	x_data, label_data = shuffle_data(x_data, original_label_data)
	return x_data, label_data


def load_gtsrb(split_train_test=True, transform=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), test_only=False, shuffle_test=True):
	"""
	Load GTSRB dataset as tensors.
	"""

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize((32, 32)),
		torchvision.transforms.ToTensor(),
		transform,
	])

	if not test_only:
		cifar_train = torchvision.datasets.GTSRB(str(utils.config.get_datasetdir(DatasetEnum.GTSRB)), download=True, split='train', transform=transform)
	cifar_test = torchvision.datasets.GTSRB(str(utils.config.get_datasetdir(DatasetEnum.GTSRB)), download=True, split='test', transform=transform)

	if not test_only:
		train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=len(cifar_train), shuffle=True)
	test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=len(cifar_test), shuffle=shuffle_test)

	x_test_, original_label_test_ = next(iter(test_loader))
	if not test_only:
		x_train_, original_label_train_ = next(iter(train_loader))

	if test_only:
		return x_test_, original_label_test_, None, None

	if split_train_test:
		return x_test_, original_label_test_, x_train_, original_label_train_

	x_data = torch.cat((x_test_, x_train_))
	original_label_data = torch.cat((original_label_test_, original_label_train_))
	x_data, label_data = shuffle_data(x_data, original_label_data)
	return x_data, label_data


def _get_huggingface_token():
	"""Resolve the Hugging Face token from common environment variable names."""
	for env_key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_TOKEN"):
		value = os.getenv(env_key)
		if value:
			return value
	return None


def _parse_imagenet_limit(env_key, fallback=None):
	value = os.getenv(env_key)
	if value is None or value.strip() == "":
		return fallback
	lowered = value.strip().lower()
	if lowered in {"none", "all", "-1", "unlimited"}:
		return None
	try:
		parsed = int(value)
	except ValueError as exc:
		raise ValueError(f"Environment variable {env_key} must be an integer or one of ['none', 'all', '-1']") from exc
	return None if parsed <= 0 else parsed


def _ensure_rgb(image):
	"""Convert PIL-style images to RGB so downstream normalization always sees three channels."""
	if hasattr(image, "mode") and hasattr(image, "convert") and image.mode != "RGB":
		return image.convert("RGB")
	return image


def _dataset_to_tensors(dataset, transform, limit=None, shuffle=False, seed=0):
	if dataset is None:
		return None, None

	total_len = len(dataset)
	indices = list(range(total_len))
	if shuffle:
		generator = torch.Generator().manual_seed(seed)
		indices = torch.randperm(total_len, generator=generator).tolist()

	if limit is not None:
		indices = indices[:min(limit, len(indices))]

	images, labels = [], []
	for idx in indices:
		example = dataset[int(idx)]
		image = example["image"]
		tensor = transform(image) if transform is not None else torchvision.transforms.ToTensor()(image)
		images.append(tensor)
		labels.append(torch.tensor(example["label"], dtype=torch.long))

	return torch.stack(images), torch.stack(labels)


def _build_subset(dataset, limit: Optional[int], shuffle: bool, seed: Optional[int]) -> torch.utils.data.Dataset:
	"""Create a deterministic subset of a torch dataset without materializing tensors."""
	total_len = len(dataset)
	if limit is None or limit >= total_len:
		return dataset
	if shuffle:
		if seed is None:
			indices = torch.randperm(total_len).tolist()
		else:
			generator = torch.Generator().manual_seed(seed)
			indices = torch.randperm(total_len, generator=generator).tolist()
	else:
		indices = list(range(total_len))
	indices = indices[:limit]
	return Subset(dataset, indices)


def _default_collate_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def load_imagenet(split_train_test=True, transform=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), test_only=False, shuffle_test=True):
	"""
	Load ImageNet-1k from the Hugging Face hub with optional sampling caps.

	Environment variables:
	  HF_TOKEN / HUGGINGFACEHUB_API_TOKEN / HUGGINGFACE_TOKEN: authentication token
	  IMAGENET_TRAIN_LIMIT: optional cap on number of training samples (default 10000)
	  IMAGENET_VAL_LIMIT: optional cap on number of validation samples (default 10000)
	"""
	try:
		from datasets import load_dataset
	except ImportError as exc:
		raise ImportError("The 'datasets' package is required to load ImageNet. Install it via 'pip install datasets'.") from exc

	token = _get_huggingface_token()
	if not token:
		raise RuntimeError("ImageNet loading requires a Hugging Face token. Set 'HF_TOKEN' (e.g. export HF_TOKEN=hf_your_token) before calling load_imagenet().")

	cache_dir = os.getenv("HF_DATASETS_CACHE")
	if not cache_dir:
		base_tmp = os.getenv("XAIBACKDOORS_TMPDIR", "/mnt/sdz/abka03_data/data") 
		cache_dir = os.path.join(base_tmp, "hf_datasets")
	try:
		os.makedirs(cache_dir, exist_ok=True)
	except OSError:
		cache_dir = os.path.join("/tmp", "hf_datasets")
		os.makedirs(cache_dir, exist_ok=True)

	load_kwargs = {"token": token, "cache_dir": cache_dir}

	try:
		imagenet_val = load_dataset("ILSVRC/imagenet-1k", split="validation", **load_kwargs)
		imagenet_train = None if test_only else load_dataset("ILSVRC/imagenet-1k", split="train", **load_kwargs)
	except Exception as exc:
		message = (
			"Failed to download ImageNet from Hugging Face. Ensure the token in 'HF_TOKEN' or 'HUGGINGFACEHUB_API_TOKEN' has access "
			"and that you accepted the dataset license at https://huggingface.co/datasets/ILSVRC/imagenet-1k."
		)
		raise RuntimeError(message) from exc
	image_size = (224, 224)
	transform_steps = [
		torchvision.transforms.Lambda(_ensure_rgb),
		torchvision.transforms.Resize(image_size),
		torchvision.transforms.ToTensor(),
	]
	if transform is not None:
		transform_steps.append(transform)
	composed_transform = torchvision.transforms.Compose(transform_steps)

	train_limit = _parse_imagenet_limit("IMAGENET_TRAIN_LIMIT", fallback=10000)
	val_limit = _parse_imagenet_limit("IMAGENET_VAL_LIMIT", fallback=10000)

	x_test_, label_test_ = _dataset_to_tensors(imagenet_val, composed_transform, limit=val_limit, shuffle=shuffle_test, seed=200)

	if test_only:
		return x_test_, label_test_, None, None

	x_train_, label_train_ = _dataset_to_tensors(imagenet_train, composed_transform, limit=train_limit, shuffle=True, seed=100)

	if split_train_test:
		return x_test_, label_test_, x_train_, label_train_

	x_data = torch.cat((x_test_, x_train_))
	original_label_data = torch.cat((label_test_, label_train_))
	x_data, label_data = shuffle_data(x_data, original_label_data)
	return x_data, label_data


class _HFImageNetDataset(torch.utils.data.Dataset):
	"""Thin wrapper to expose Hugging Face ImageNet split as a torch Dataset."""
	def __init__(self, hf_dataset, transform, limit: Optional[int], shuffle: bool, seed: Optional[int]):
		self._hf_dataset = hf_dataset
		self._transform = transform
		indices = list(range(len(hf_dataset)))
		if shuffle:
			if seed is None:
				indices = torch.randperm(len(hf_dataset)).tolist()
			else:
				generator = torch.Generator().manual_seed(seed)
				indices = torch.randperm(len(hf_dataset), generator=generator).tolist()
		if limit is not None:
			indices = indices[:min(limit, len(indices))]
		self._indices = indices

	def __len__(self):
		return len(self._indices)

	def __getitem__(self, item):
		example = self._hf_dataset[self._indices[item]]
		image = _ensure_rgb(example["image"])
		tensor = self._transform(image)
		label = torch.tensor(example["label"], dtype=torch.long)
		return tensor, label


def load_data_loaders(
		dataset: DatasetEnum,
		train_batch_size: int,
		test_batch_size: Optional[int] = None,
		train_limit: Optional[int] = None,
		test_limit: Optional[int] = None,
		test_only: bool = False,
		shuffle_train: bool = True,
		shuffle_test: bool = False,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
	"""Return DataLoader objects instead of materializing full tensors."""
	if test_batch_size is None:
		test_batch_size = train_batch_size

	if dataset == DatasetEnum.CIFAR10:
		transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		train_dataset = None if test_only else torchvision.datasets.CIFAR10(str(utils.config.get_datasetdir(DatasetEnum.CIFAR10)), download=True, train=True, transform=transform)
		test_loader = None
		if test_limit != 0:
			test_dataset = torchvision.datasets.CIFAR10(str(utils.config.get_datasetdir(DatasetEnum.CIFAR10)), download=True, train=False, transform=transform)
			test_subset = _build_subset(test_dataset, test_limit, shuffle_test, seed=200)
			if len(test_subset) > 0:
				test_loader = _default_collate_loader(test_subset, test_batch_size, shuffle_test)
		train_subset = None if train_dataset is None else _build_subset(train_dataset, train_limit, shuffle_train, seed=None)
		train_loader = None if train_subset is None else _default_collate_loader(train_subset, train_batch_size, shuffle_train)
		return train_loader, test_loader

	if dataset == DatasetEnum.CIFAR100:
		transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
		])
		train_dataset = None if test_only else torchvision.datasets.CIFAR100(str(utils.config.get_datasetdir(DatasetEnum.CIFAR100)), download=True, train=True, transform=transform)
		test_loader = None
		if test_limit != 0:
			test_dataset = torchvision.datasets.CIFAR100(str(utils.config.get_datasetdir(DatasetEnum.CIFAR100)), download=True, train=False, transform=transform)
			test_subset = _build_subset(test_dataset, test_limit, shuffle_test, seed=200)
			if len(test_subset) > 0:
				test_loader = _default_collate_loader(test_subset, test_batch_size, shuffle=shuffle_test)
		train_subset = None if train_dataset is None else _build_subset(train_dataset, train_limit, shuffle_train, seed=None)
		train_loader = None if train_subset is None else _default_collate_loader(train_subset, train_batch_size, shuffle_train)
		return train_loader, test_loader

	if dataset == DatasetEnum.GTSRB:
		transform = torchvision.transforms.Compose([
			torchvision.transforms.Resize((224, 224)),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		train_dataset = None if test_only else torchvision.datasets.GTSRB(str(utils.config.get_datasetdir(DatasetEnum.GTSRB)), download=True, split='train', transform=transform)
		test_loader = None
		if test_limit != 0:
			test_dataset = torchvision.datasets.GTSRB(str(utils.config.get_datasetdir(DatasetEnum.GTSRB)), download=True, split='test', transform=transform)
			test_subset = _build_subset(test_dataset, test_limit, shuffle_test, seed=200)
			if len(test_subset) > 0:
				test_loader = _default_collate_loader(test_subset, test_batch_size, shuffle_test)
		train_subset = None if train_dataset is None else _build_subset(train_dataset, train_limit, shuffle_train, seed=None)
		train_loader = None if train_subset is None else _default_collate_loader(train_subset, train_batch_size, shuffle_train)
		return train_loader, test_loader

	if dataset == DatasetEnum.IMAGENET:
		try:
			from datasets import load_dataset
		except ImportError as exc:
			raise ImportError("The 'datasets' package is required to load ImageNet. Install it via 'pip install datasets'.") from exc

		token = _get_huggingface_token()
		if not token:
			raise RuntimeError("ImageNet loading requires a Hugging Face token. Set 'HF_TOKEN' before calling load_data_loaders().")

		cache_dir = os.getenv("HF_DATASETS_CACHE")
		if not cache_dir:
			base_tmp = os.getenv("XAIBACKDOORS_TMPDIR", "/mnt/sdz/abka03_data/data")
			cache_dir = os.path.join(base_tmp, "hf_datasets")
		try:
			os.makedirs(cache_dir, exist_ok=True)
		except OSError:
			cache_dir = os.path.join("/tmp", "hf_datasets")
			os.makedirs(cache_dir, exist_ok=True)

		load_kwargs = {"token": token, "cache_dir": cache_dir}
		imagenet_val = None if test_limit == 0 else load_dataset("ILSVRC/imagenet-1k", split="validation", **load_kwargs)
		imagenet_train = None if test_only else load_dataset("ILSVRC/imagenet-1k", split="train", **load_kwargs)

		transform_steps = [
			torchvision.transforms.Lambda(_ensure_rgb),
			torchvision.transforms.Resize((224, 224)),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
		transform = torchvision.transforms.Compose(transform_steps)

		test_loader = None
		if imagenet_val is not None:
			test_dataset = _HFImageNetDataset(imagenet_val, transform, limit=test_limit, shuffle=shuffle_test, seed=200)
			if len(test_dataset) > 0:
				test_loader = _default_collate_loader(test_dataset, test_batch_size, shuffle=False)

		if imagenet_train is None:
			return None, test_loader

		train_dataset = _HFImageNetDataset(imagenet_train, transform, limit=train_limit, shuffle=shuffle_train, seed=None)
		train_loader = _default_collate_loader(train_dataset, train_batch_size, shuffle=shuffle_train)
		return train_loader, test_loader

	raise Exception(f'Unknown dataset {dataset}')
