UID := $(shell id -u)
GID := $(shell id -g)


preprocess:
	python -m src.datasets.preprocess_gen \
		--image-folder data/raw/preprocess_supervisely/images \
		--ground-truth data/raw/preprocess_supervisely/masks \
		--dst-folder data/preprocessed/supervisely \
		--extension .png

split:
	python -m src.datasets.split \
		--image-folder data/preprocessed/supervisely/images \
		--ground-truth data/preprocessed/supervisely/masks \
		--dst-train data/datasets/supervisely/train \
		--dst-val data/datasets/supervisely/val \
		--extension .png \
		--percentage 0.3

train:
	python -m src.train --config=configs/pytorch-gen.yaml
evaluate:
	python -m src.eval --config=configs/pytorch-gen.yaml

# If you are running in remote machine you need to open
# browser in machine-IP:PORT
# like http://192.168.30.10:16006
tensorboard:
	tensorboard --logdir=logs --host 0.0.0.0 --port 6006