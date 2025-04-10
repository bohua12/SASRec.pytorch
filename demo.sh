python main_new.py --dataset=ml-100k --train_dir=amazon32a --maxlen=200 --verbose=0 

git clone https://github.com/bohua12/SASRec.pytorch
cd SASRec.pytorch
git checkout CDR_linear
python main_new.py --dataset=ml-100k --train_dir=amazon32a --maxlen=200 --verbose=0  --verification_frequency=5 --lr=0.001

git clone https://github.com/bohua12/SASRec.pytorch
cd SASRec.pytorch
git checkout CDR_linear
python main_new.py --dataset=ml-100k --train_dir=amazon32a --maxlen=200 --verbose=0  --verification_frequency=5 --lr=0.005

ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDlB0dgEHCG4BwJ/bM3h6ztVIZpiIbtaKrXLUzbToS67