Steps to follow to run the GLUECoS pipeline :

    Requirements :
        First, manually install torch, specific to your platform
        Then run pip install -r requirements.txt if you only want to download data in the processed format.
        Run pip install -r requirements.txt (in the Code folder) if you want to evaluate bert/xlm/xlmr model variations on the GLUECoS tasks.

    
    Twitter developer account :
        You will need a twitter developer account to download the twitter datasets. Enter your consumer key (API key); consumer secret key (API secret key); access token and access secret token, one per line in the same order, in twitter_authentication.txt (without any other keywords)
    
    Download data :
        Run the below command to download the original datasets and process them to bring them in the required format. They will be stored in the Processed_Data folder. Enter your azure subscription key as an argument to the script below to download transliterations as well. Instructions to procure your subscriptio key can be found here.
        The Original_Data folder contains the ID splits for datasets that do not have a pre-defined split, which is used in the preprocessing. Note that the test labels are not gold labels. They have been assigned a separate token to maintain fairness in the benchmarking.
            ./download_data.sh SUBSCRIPTION_KEY

    Fine-tune models for prediction :
        The code contains 4 different evaluation scripts, one each for token level tasks (LID(en_es/en_hi), NER(en_es/en_hi), POS(en_es/en_hi_fg/en_hi_ud)); sentence level tasks (Sentiment(en_es/en_hi)); QA tasks (QA(en_hi)) and NLI tasks (NLI(en_hi)). Run the below command to fine-tune your model on any of the task. The models currently supported are bert, xlm and xlmr based models.

        bash train.sh MODEL MODEL_TYPE TASK 
        Example Usage :
            
            bash train.sh bert-base-multilingual-cased bert POS_EN_HI_FG

        You can also run fine-tuning for all tasks with the following command :

            bash train.sh bert-base-multilingual-cased bert ALL

   