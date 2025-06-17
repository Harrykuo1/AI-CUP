.PHONY: pseudo train test

CONFIG=config/config.yaml
PSEUDO=pseudo/pseudo_task.csv
BREAK_EPOCH=3
BREAK_FOLD=1

all: pseudo train test

install:
	conda env create -f enviroment.yaml
	conda init
	conda activate AICUP
	pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pseudo:
	python gen_pseudo.py --config $(CONFIG)

train:
	python train.py --config $(CONFIG) --pseudo_csv $(PSEUDO)

train_break:
	python train.py --config $(CONFIG) --pseudo_csv $(PSEUDO) --break_epoch $(BREAK_EPOCH) --break_fold $(BREAK_FOLD)

test:
	python test_top.py --config $(CONFIG) --top_frac 0.8
clean:
	rm weight/* log/auc/* log/acc/* log/loss/*