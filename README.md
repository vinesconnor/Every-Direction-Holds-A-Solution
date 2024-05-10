# Every Direction Holds a Solution
For full implementations and pipelines ($\beta$ fitting -> modeling -> naive inference), see the 3 notebooks labeled Main. For the two custom ensemble approaches using MLPs, see the two files labeled Ensemble.

Each notebook has a cell for hyperparameters at the beginning. We provide default values but feel free to change them. `Solver` controls which solver to use for sklearn's Logistic Regression module. Typically, liblin (`solver = True`) is better, but it is computationally slower. So we set the default value to `False`.

Note: Figures in the notebooks may look off since we ran them with very low values (like `N = 5`) to test the code after transporting it from Colab.
