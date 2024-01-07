UID := $(shell id -u)
GID := $(shell id -g)


preprocess:
	python -m src.datasets.preprocess_gen \
		--image-folder data/raw/Human-Segmentation-Dataset-master/train-images \
		--ground-truth data/raw/Human-Segmentation-Dataset-master/train-ground-truth \
		--dst-folder data/preprocessed/train \
		--extension .jpg

	python -m src.datasets.preprocess_gen \
		--image-folder data/raw/Human-Segmentation-Dataset-master/val-images \
		--ground-truth data/raw/Human-Segmentation-Dataset-master/val-ground-truth \
		--dst-folder data/preprocessed/validation \
		--extension .jpg

train:
	python -m src.train --config=configs/pytorch-gen.yaml