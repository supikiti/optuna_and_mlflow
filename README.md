## 概要
輪講用に作成した Optuna と mlflow 用のチュートリアルコードを載せています．

## 環境
```bash
Python 3.6.5

mlflow==1.12.1
torch==1.7.0
torchvision==0.8.1
optuna==2.3.0
```

## 使用例

```bash
pip install optuna torch torchvision mlflow
git clone https://github.com/supikiti/optuna_and_mlflow.git
cd optuna_and_mlflow
python optuna_cnn_mlflow.py
mlflow ui
```

### ローカルPC上で実行する場合
```bash
open http://127.0.0.1:5000
```

### ssh先のサーバー上で実行する場合
```bash
ssh (ここは適宜) -L 9000:localhost:5000
open http://localhost:9000
```

## 参考にしたコード

### 公式Tutorial(Pytorch)
- https://github.com/optuna/optuna/blob/master/examples/pytorch_simple.py
### その他
- https://gist.github.com/nogawanogawa/f6b5a36143a56c62c8d100ebf85702cb
- https://github.com/Yushi-Goto/optuna-with-pytorch
