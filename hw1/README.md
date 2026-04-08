# 多標籤文字分類任務 (COVID-19 Tweets Multi-label Classification)

本專案支援兩種不同的深度學習架構來進行推文的多標籤分類，分別是原本輕量基礎的 LSTM 架構，與這次新加入基於 Transformer 的 COVID-Twitter-BERT 架構：

1. **LSTM + Word2Vec (GloVe-Twitter)**
2. **BERT (COVID-Twitter-BERT-v2)**

## 環境要求
執行本專案前，請確保安裝了requirements.txt中的套件

## 第一套模型：LSTM (您原初實作)
- 利用 `module.py` 定義雙向多層 LSTM 架構
- 使用 `gensim` 下載的 `glove-twitter-100` 預訓練詞向量做為初始詞嵌入。
- **訓練指令**：
  ```bash
  python train.py
  ```
  模型表現進步時將自動儲存權重至 `best_model.pth`。
  
- **預測指令**：
  ```bash
  python test.py
  ```
  將會產出可用的 `submission.csv`。

---

## 第二套模型：COVID-Twitter-BERT (全新擴增)
- 利用 Hugging Face 的 `transformers` 套件，直接引入專為新冠肺炎推文設計的最佳預訓練模型 `digitalepidemiologylab/covid-twitter-bert-v2`。
- 自動呼叫獨立的 Dataset Loader (`dataset_bert.py`)，進行 BERT 最佳化的 Padding、Truncation 與 Attention Mask。
- **訓練指令**：
  ```bash
  python train_bert.py
  ```
  程式將自動把表現最好的模型權重存至 `best_bert_model.pth`。
  
- **預測指令**：
  ```bash
  python test_bert.py
  ```
  腳本將去 `best_bert_model.pth`抓取權重，並產出`submission_bert.csv` 

## 資料來源格式 (HW1_dataset)
資料為 JSON 格式（包含在 `HW1_dataset` 資料夾），具備對應的 `ID`、主要推文內容 `tweet`，與對應的標籤群組 `labels`。
