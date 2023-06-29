# WikiFact Dataset

## Download dataset:

https://drive.google.com/file/d/1bdwjmnPIZEObNh1T-YTXH4HCAVcK_AaR/view?usp=drive_link

## Environment:

```
pip install -r requirements.txt
```

## Evaluation：

1. Put the dataset in  `WikiFact/data`;

2. Set your **openai api_key** in `wikifact_chatgpt.py`  `wikifact_002.py`  `wikifact_003.py` **L6**, your Claude **userOAuthToken** and **channel_id** in `test_wikifact.sh` **L9**, and your **model-path** in `test_wikifact.sh ` **L12-24**;

3. To get the evaluation scores of all models:

   ```
   bash test_wikifact.sh
   ```

4. All generated outputs are saved in `wikifact/generation`.