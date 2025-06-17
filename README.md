# 桌球智慧球拍資料的精準分析 - AI CUP 2025春季賽

## 環境安裝

### 使用Conda安裝
```bash
conda env create -f enviroment.yaml
conda activate AICUP
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

## 專案結構
專案結構如下，請將data放置於以下正確位置

```
project/
├── README.md                    # 本文件
├── enviroment.yaml             # Conda環境配置
├── 3090.yaml                   # 訓練配置檔
├── dataloader.py              # 資料載入與預處理
├── model_large.py             # 模型架構定義
├── train.py                   # 訓練主程式
├── test_top.py                # 測試與推論
├── gen_pseudo.py              # 偽標籤生成
├── plotter.py                 # 視覺化工具
├── data/                      # 資料目錄
│   ├── train/
│   │   ├── train_info.csv
│   │   └── train_sensors/
│   └── test/
│       ├── test_info.csv
│       └── test_sensors/
├── weight/                    # 模型權重存放
├── log/                      # 訓練日誌與圖表
├── result/                   # 實驗結果
└── pseudo/                   # 偽標籤資料
```

## 檔案說明

| 檔案名稱 | 功能說明 |
|---------|---------|
| `config.yaml` | 訓練配置檔案，包含所有超參數設定 |
| `dataloader.py` | 資料載入與預處理，包含特徵提取和批次處理 |
| `model_large.py` | 模型架構定義，包含多分支CNN和Transformer |
| `train.py` | 訓練主程式，執行交叉驗證和模型訓練 |
| `test_top.py` | 測試推論程式，生成最終提交檔案 |
| `plotter.py` | 訓練過程視覺化工具 |
| `enviroment.yaml` | Conda環境配置檔 |

## 使用方法

### 1. 訓練模型
```bash
make train
```

### 2. 生成預測結果
```bash
make test
```

## 輸出檔案
- 模型權重：`weight/fold{X}_s{Y}.pth`
- 訓練日誌：`log/logs.txt`
- 學習曲線：`log/loss/`, `log/acc/`, `log/auc/`
- 最終提交：`submission_top_models.csv`

## 配置
- Python 3.12.3
- CUDA 12.4 (GPU訓練)
- GPU：兩張 NVIDIA RTX 3090
