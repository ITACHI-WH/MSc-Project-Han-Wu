step1, run the command 'python video_process.py' on the linux server to execute encoding, decoding & play steps for all four encoding formats and all videos
Step2: After about 1 week, step1 will finish and generate the record 'out.csv'
Step3: run the command 'python read_data_from_db.py' in the local computer to download the power monitoring data from database, please mind to change the saving data slice in read_data_from_db.py
Step4: the reuslts of step3 is stored in the 'power.csv'
Step5: run command 'python analysis_data.py' to generate the figures and execute machine learning based algorithm
Step6: run command 'python gui.py' to open the GUI and you can fill the features value from to predict the best encoding formats