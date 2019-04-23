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
    testAF\ (optional)
      clas1\
      clas2\
      ...

Running train.py:
  If the above path is adhered to, no additional steps are needed to run train.py.

Running test.py:
  In the main file, set the easy parameter to 1 if you wish to use the testAF folder in which case you would put the easy test files in that folder for convenience. Otherwise you can set the easy parameter to 0 and only utilize the test folder for all your test files.

All resulting files, including model and text files, will be generated in the working directory of the project.