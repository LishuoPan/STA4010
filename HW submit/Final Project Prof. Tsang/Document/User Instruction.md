#### User Instruction:

##### Detail of Analysis and framework design:

- under "Document" folder "Document.pdf"

##### Dependency package:

- [NumPy](http://www.numpy.org/): matrix computation.

- [Pandas](https://pandas.pydata.org/): data reader.

- [scikit-learn](https://scikit-learn.org/): contain basic machine learning function modules.

  These basic packages should be contained in [Anaconda](https://www.anaconda.com/). Or any basic python environment setups.

  Otherwise, use conda or pip to install them before run the `main.py`

##### How to run

- put the "Day5.csv" .csv file under the "testing_data" folder.

- Use python to run `main.py`. Newest Python is preferred.

- Basic tuning: 

  - in the main.py line 98,99. 
  - Default: SafeGaurd = 0; Split = 5(High Risk). 
  - If lose money, you can change it to low risk: SafeGaurd = 0; Split = 10(Low Risk) or even SafeGaurd = 1; Split = 10(Low Risk). The detail is in the document.

- *Advanced tuning ( *Not try this, if not familiar with hyper-parameters I introduced):

  - in the TradingModel.py. Chang the hyper-parameters in the def \__init__ part according to document.




##### If you encountered any problem please contact me:

E-mail: 115010208@link.cuhk.edu.cn

WeChat: panlishuo001



