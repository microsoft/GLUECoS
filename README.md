# GLUECoS

Steps to follow to run the GLUECoS pipeline :

1. Requirements :
    First, manually install torch, specific to your platform
    Then run pip install -r requirements.txt if you only want to download data in the processed format.
    Run pip install -r requirements.txt (in the Code folder) if you want to evaluate bert/xlm/xlmr model variations on the GLUECoS tasks.

    
2. Twitter developer account :
    You will need a twitter developer account to download the twitter datasets. Enter your consumer key (API key); consumer secret key (API secret key); access token and access secret token, one per line in the same order, in twitter_authentication.txt (without any other keywords)
    
3. Download data :
    Run the below command to download the original datasets and process them to bring them in the required format. They will be stored in the Processed_Data folder. Enter your azure subscription key as an argument to the script below to download transliterations as well. Instructions to procure your subscriptio key can be found here.
    The Original_Data folder contains the ID splits for datasets that do not have a pre-defined split, which is used in the preprocessing. Note that the test labels are not gold labels. They have been assigned a separate token to maintain fairness in the benchmarking.
    ```
        ./download_data.sh SUBSCRIPTION_KEY
    ```

4. Fine-tune models for prediction :
    The code contains 4 different evaluation scripts, one each for token level tasks (LID(en_es/en_hi), NER(en_es/en_hi), POS(en_es/en_hi_fg/en_hi_ud)); sentence level tasks (Sentiment(en_es/en_hi)); QA tasks (QA(en_hi)) and NLI tasks (NLI(en_hi)). Run the below command to fine-tune your model on any of the task. The models currently supported are bert, xlm and xlmr based models.

    ```
    bash train.sh MODEL MODEL_TYPE TASK 
    ```
    Example Usage :
    ```    
    bash train.sh bert-base-multilingual-cased bert POS_EN_HI_FG
    ```
    You can also run fine-tuning for all tasks with the following command :
    ```
    bash train.sh bert-base-multilingual-cased bert ALL
    ```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
