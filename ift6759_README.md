The files relevant for the course ift6759 all start by its name. Here is a quick description of what is in each file:
* ift6759_train_Glow_2mTemp.py: This is the main experiment. To run, just use: python ift6759_train_Glow_2mTemp.py
* ift6759_bcsd.py: This is an attempt at using the author's code for a standard downscaling method called BCSD which didn't yield any sensical results on time. To run, just use: python ift6759_bcsd.py

The virtual environement can be generated via conda using environment.yml. But to run experiments, you need to have our preprocessed data on disk, which is available under request at: https://drive.google.com/drive/folders/1B32SUh9NFyyUHoiG1jzHMzGUt9EtD470?usp=sharing

As mentionned in the final report, the contribution in terms of lines of code is limited because most of our time went towards figuring out the environment dependencies and debugging the original code in order to make it work.