There's a lot of details about the project in the [ReadMe](https://github.com/Leci37/stocks-prediction-Machine-learning-RealTime-telegram/blob/master/README.md) or [Luis' Medium blog](https://medium.com/@LuisCasti33/lectrade-machine-learning-with-technical-patterns-and-real-time-alerts-for-buying-and-selling-in-b4ecc59b29cb) make sure you read it.

### Background 
Welcome to the setup guide for the project. Here, you will find step-by-step instructions to set up the project on your system. The guide assumes you are using Windows, but the steps are similar for Mac OS and Linux.

#### Pre-requisite 
Before you begin, please make sure you have the following prerequisites:

* [Anaconda Python](https://www.anaconda.com/download) (recommended) or Python version < 3.10
* [Visual C++](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170) installed for Tensorflow on Windows
* An IDE of your choice, such as [PyCharm](https://www.jetbrains.com/pycharm/download/) or [Visual Studio Code](https://code.visualstudio.com/download)
* Clone this repository
* Optional step - Setup your python virtual environment

## Step by Step

### Package Installation
1. Installing the **TA-Lib** Python package may be slightly tricky. You can try the instructions provided on their [GitHub](https://ta-lib.github.io/ta-lib-python/install.html) or directly from their [source](https://ta-lib.github.io/ta-lib-python/install.html).  If you are a Windows user, you can also download the version applicable to your Python from the [University of California, Irvine archive](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib).
2. Next, depending on your system (Python or Python3), run the following command to install the required Python packages (from the project directory):

   ```python -m pip install -r requirements.txt``` 
  
   Note: If you encounter a warning or a message saying "tensorflow-gpu" couldn't be installed, you can ignore it for now.
If there were no major errors, you should have everything installed and ready to go.
### Configuration
Before running the project, you need to configure a few settings:
1. Setting up Stock Tickers - Open the file  ```_KEYS_DICT.py``` and go to line 13. You will see a line like this:
  ```["UBER", "PYPL"],```
   
   Here, you can enter the stock tickers you are interested in from NASDAQ or DOW Jones.
#### Optional: GPU Configuration
If you have a suitable GPU that you wish to use for computation, find line 62 in ```_KEYS_DICT.py``` and change the value of ```USE_GPU``` to ```"Yes"```.
Additionally, on line 63 of ```_KEY_DICT.py``` you can adjust the ```PER_PROCESS_GPU_MEMORY_FRACTION``` value.  This step is optional but can be useful if you only want to use a fraction of your GPU resource.
2. Setting up Telegram Bot - To set up your own Telegram Bot, follow the [tutorial](https://www.siteguarding.com/en/how-to-get-telegram-bot-api-token) 
Once you have obtained the Bot token, your admin user ID, and the recipient's ID number, update the credentials in the file  ```ztelegram_send_message_handle.py```. 
3. Setting up Twitter Handle - If you want to set up a Twitter handle, you can raise an issue on GitHub or reach out through the details given in the [README](https://github.com/Leci37/stocks-prediction-Machine-learning-RealTime-telegram/blob/master/README.md)  file for ```twi_credential.py```

### Running the Project
1. If you have followed the above steps correctly, you should now be able to run the project files. Start by running the files named from 0 to 5 in sequential order.

    ![imp_files](/readme_img/imp_files.PNG "Run in sequence")
   
   If you encounter any missing files, refer to the [README](https://github.com/Leci37/stocks-prediction-Machine-learning-RealTime-telegram/blob/master/README.md) file for further instructions.

### Known Issues
If you encounter an error similar to the one shown below, follow the steps below; otherwise, it is not required:
```commandline
Traceback (most recent call last):
  File "C:\Users\user\projects\stocks-prediction-Machine-learning-RealTime-telegram\get_technical_indicators.py", line 46, in <module>
    df_download = yhoo_history_stock.get_favs_SCALA_csv_stocks_history_Download_list(list_stocks, CSV_NAME, opion, GENERATED_JSON_RELATIONS = GENERATED_JSON_RELATIONS)
  ...
AttributeError: 'Series' object has no attribute 'append'. Did you mean: '_append'?
```

1. Go to the location of your virtual environment. For example:


    File "C:\Users\user\projects\virtualenv\stocks-prediction-Machine-learning-RealTime-telegram\lib\site-packages\pandas_ta\overlap\mcgd.py", line 24, in mcgd
    mcg_ds = close[:1].append(mcg_cell[1:])

2. Make a copy of the file ```pandas_ta\overlap\mcgd.py```

3. Change this line from 

   ```mcg_ds = close[:1].append(mcg_cell[1:])``` to ```mcg_ds = close[:1]._append(mcg_cell[1:])```
Notice the addition of the underscore before append.