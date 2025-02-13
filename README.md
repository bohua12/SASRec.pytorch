## 📌 Introduction
This repo my attempt to improve on the [pytorch implementation](https://github.com/pmixer/SASRec.pytorch), which itself is modified based on the [paper author's tensorflow implementation](https://github.com/kang205/SASRec). The incumbent switches to PyTorch(v1.6) for simplicity, fixed issues like positional embedding usage etc. (making it harder to overfit, except for that, in recsys, personalization=overfitting sometimes)

## 🔥 What's New in This Version?

🏋️ **Added Weight Decay** – Introduced for better regularization, with a default 1e-3. This can be adjusteted via a new hyperparameter ```verification_frequency```

⏳ **More Frequent Validation** – Reduced validation frequency from 20 to a default 5, making training feedback more responsive. This can be adjusted via a new hyperparameter ```weight_decay```

## 🚧 In Progress:

📦 **Using DataLoader** – Switched to PyTorch’s DataLoader for more efficient data batching and loading.

🔄 **Updated Validation** – Improved the validation strategy for better model assessment.

🎛 **Hyperparameter Tuning** – Optimize model performance.

⏹ **Early Stopping** – Stop the model when overfitting is detected, to prevent unnecessary overfitting and save computation time.

📊 **Model Evaluation** – Evaluation metrics for better insights and comparison of model performance.


## 📜 Taken from the [pytorch implementation](https://github.com/pmixer/SASRec.pytorch)
This repository enhances the previous PyTorch implementation (pmixer/SASRec.pytorch) with the following improvements:
to train:

```
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```

just inference:

```
python main.py --device=cuda --dataset=ml-1m --train_dir=default --state_dict_path='ml-1m_default/SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --maxlen=200

```

output for each run would be slightly random, as negative samples are randomly sampled, here's my output for two consecutive runs:

```
1st run - test (NDCG@10: 0.5897, HR@10: 0.8190)
2nd run - test (NDCG@10: 0.5918, HR@10: 0.8225)
```

pls check paper author's [repo](https://github.com/kang205/SASRec) for detailed intro and more complete README, and here's the paper bib FYI :)

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

I see a dozen of citations of the repo recently🫰, here's the repo bib if needed.
```
@software{Huang_SASRec_pytorch,
author = {Huang, Zan},
title = {PyTorch implementation for SASRec},
url = {https://github.com/pmixer/SASRec.pytorch},
year={2020}
}
```
