The code is adapted to work in SLURM environment.
If you want to run it on a local environment, you can run the main script and adapt the main function.
The main script include a function for each experiment, and you should choose what experiment you want to run by uncomment the desired experiment (and set the function's parameter as you want).
Each experiment produce a csv file including the p-values of each model and other details. All csvs can be analyzed in the analyze.ipynb script.

Attached a virtual environment file MRD.yml with all the packages that required to run this code. 