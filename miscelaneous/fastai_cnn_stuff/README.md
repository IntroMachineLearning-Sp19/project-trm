TEAM TRM

PROJECT MEMBERS: TRUNG, NIKITA, ROSEMOND & MICHAEL

Initially, our python files MUST be run in the order of train.py and then test.py. Train.py will generate a model file and save it to the working directory after it is run. After you are happy with the trained network only the test.py will be utilized. Train.py and test.py MUST be in the same directory.

Before running train.py:
  Create this directory structure in the same path as the train.py and test.py:

  data\
    train\
      clas1\
      clas2\
      ...
    valid\
      clas1\
      clas2\
      ...
    test\
      clas1\
      clas2\
      ...
    models\
    testAF\
      clas1\
      clas2\
      ...

    Place the easy test data in the testAF  directory in the hard test data in the test directory

    Warning:
      Please close your file explorer after placing the files in the necessary directories to avoid error.

Running train.py:
  If the above path is adhered to, no additional steps are needed to run train.py.

Running test.py:
  If the above path is adhered to, no additional steps are needed to run test.py.

Results:
  The predictions will be in an estimatedEasyLabels.txt file and estimatedHardLabels.txt file accordingly.
