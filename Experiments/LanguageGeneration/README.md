# LAMBADA Dataset

## Download dataset:

https://drive.google.com/file/d/1HHDR1nMYoHCz5FNrppv3oee9E4CWZZUL/view?usp=drive_link

## Environment:

```
pip install -r requirements.txt
```

## Evaluation：

1. Put the dataset in  `LAMBADA/data`;

2. Set your **openai api_key** in `lambada_chatgpt.py`  `lambada_002.py`  `lambada_003.py` **L6**, your Claude **userOAuthToken** and **channel_id** in `test_lambada.sh` **L6**, and your **model-path** in `test_lambada.sh ` **L7-13**;

3. To get the evaluation scores of all models:

   ```
   bash test_lambada.sh
   ```

4. All generated outputs are saved in `LAMBADA/generation`.





# WMT22 Dataset

## Download dataset:

https://drive.google.com/file/d/1p2jG-h8NTUOCh-X_rIZHjV_wjzfGWDE5/view?usp=drive_link

## Environment:

```
pip install -r requirements.txt
```

## Evaluation：

1. Put the dataset in  `WMT22/data`;

2. Set your **openai api_key** in `wmt_chatgpt.py`  `wmt_002.py`  `wmt_003.py` **L6**, your Claude **userOAuthToken** and **channel_id** in `test_wmt.sh` **L9**, and your **model-path** in `test_wmt.sh ` **L11-23**;

3. To get the evaluation scores of all models:

   ```
   bash test_wmt.sh
   ```

4. All generated outputs are saved in `WMT22/generation`.





# XSum Dataset

## Download dataset:

https://drive.google.com/file/d/1SI7dafnXvcp96nQzhgMgafOso0XUmGFo/view?usp=drive_link

## Environment:

```
pip install -r requirements.txt
```

## Evaluation：

1. Put the dataset in  `XSum/data`;

2. Set your **openai api_key** in `xsum_chatgpt.py`  `xsum_002.py`  `xsum_003.py` **L9**, your Claude **userOAuthToken** and **channel_id** in `test_xsum.sh` **L13**, and your **model-path** in `test_xsum.sh ` **L11-23**;

3. To get the evaluation scores of all models:

   ```
   bash test_xsum.sh
   ```

4. All generated outputs are saved in `XSum/generation`.





# HumanEval Dataset

## Download dataset:

https://drive.google.com/file/d/1fy3LorG0TkHNK22rlAYV1yftgU4CAipx/view?usp=drive_link

## Environment:

```
pip install -r requirements.txt
```

## Evaluation：

1. Put the dataset in  `HumanEval/data`;

2. Set your **openai api_key** in `model.py`  **L267**  `util.py` **L6**, your Claude **userOAuthToken** and **channel_id** in `model.py` **L112-113**, and your **model-path** in `test_humaneval.sh ` **L10-22**;

3. To get the evaluation scores of all models:

   ```
   bash test_humaneval.sh
   ```

4. All generated outputs are saved in `HumanEval/generation`.