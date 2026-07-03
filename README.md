# Predicting the Nations Champtionship 2026

This project utilises Bayesian hierachical modelling to model the upcoming nations championship using PyMC.

Modelling htis tournament presented some interesting challenges. Testing and fitting modells proved difficult due to a lack of data, in part down to the relative sparsity of international rugby fixtures, and quality of data sources. As this is the first antions championship, previous competitions could not be relied on. Rugby itself presents an interesting modelling problem, with different ways of scoring (tries, penalty tries, conversions, penalties, drop goals), increasing the complexity compared to sports like football. I found that, somewhat counter intuitvely, models needed to be more complex to imporve interpretability. There is certainly room for inprovement here. I would have liked time to to delve into a few different model structures to better predict capture high scoring games (perhaps with a team interaction), probe a bit more into the sensitivity to temporal weighting parameters (and how to select these), and figure out a better way to model penalties. That being said, I should probably get back to working on my PhD.

I have made a few different classes which implement different model types. They are all subclasses of RugbyModel, which implements the evaluation, simulation, plotting etc. Hopefully this will make it easier to add new models in the future, or for anyone to adapt the code for their own models. 


`rugbyModels.py` contains the RugbyModel class with helper functions to simulate the tournament and plot findings along with some model stucture subclasses which can be used to fit different model types. 

For model selection see `model_comparison.ipynb`. For the tournament simulation and preiction (the fun stuff) see `predictions.ipynb`.

## Requirements
    - numpy 
    - matplotlib
    - pandas
    - pymc 
    - arviz
    - seaborn
    - scipy


# Come on Ireland!
