# Bayesian explorations

This repositories contains examples, tutorials and small projects with two common themes: Bayesian data analysis (broadly speaking) and [Tensorflow Probability (TFP)](https://www.tensorflow.org/probability/overview) as a tool to do that.

## Some Python libraries for Bayesian models

Here's a (surely incomplete) list of Python libraries to build Bayesian models:
- [Tensorflow Probability (TFP)](https://www.tensorflow.org/probability/overview).

- [PyMC](https://docs.pymc.io/en/latest/learn/examples/pymc_overview.html) (version 4.0 is in pre-release as of February 2022).

- [NumPyro](https://num.pyro.ai/en/latest/index.html#introductory-tutorials) (a lightweight version of [Pyro](https://pyro.ai/examples/index.html) relying on a NumPy backend that uses JAX instead of a PyTorch backend).

- [PyStan](https://pystan.readthedocs.io/en/latest/) (the Python interface to the Stan platform, written in C++).

- [ArviZ](https://arviz-devs.github.io/arviz/index.html) (a library for exploratory analysis of Bayesian models, e.g. plotting, diagnostics and model comparison, that works on top of many packages, among which PyMC, PyStan and TFP).

- [Edward](http://edwardlib.org/).

**Suggestion:** start with one and focus on that. I've personally chosen **TFP** because I find it quite complete in terms of features and because being from Google we can be sure it's well maintained - plus, I was curious about exploring TensorFlow in general and Bayesian neural networks in particular.

## Learning resources

Books:
- Davidson-Pilon (and contributors) - [Probabilistic Programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers): very good starting point, very hands-on, with all the content freely available in notebooks on GitHub and coding done in PyMC, TFP and for some chapters Pyro as well.

- Gelman, Carlin, Stern, Dunson, Vehtari, Rubin - [Bayesian data analysis](http://www.stat.columbia.edu/~gelman/book/): one of the bibles on Bayesian models, with Gelman being one of the world's leading scientists on the subject. Coding in PyMC is also available (see below).

- McElreath - [Statistical rethinking](https://www.routledge.com/Statistical-Rethinking-A-Bayesian-Course-with-Examples-in-R-and-STAN/McElreath/p/book/9780367139919): another very famous book, with an [associated GitHub repo](https://github.com/rmcelreath/rethinking) and lecture course ([here](https://github.com/rmcelreath/stat_rethinking_2022) for year 2022) with YouTube videos (see the playlist linked therein). Coding in PyMC is also available (see below).

- Kruschke - [Doing Bayesian data analysis](https://sites.google.com/site/doingbayesiandataanalysis/): another good book on Bayesian models.

- Martin, Kumar, Lao - [Bayesian analysis with Python - Second edition](https://www.packtpub.com/product/bayesian-analysis-with-python-second-edition/9781789341652): from some of the authors of PyMC and TFP, good book on Bayesian models, covers topics like model assessment and comparison in the Bayesian setting. Examples are given in PyMC and TFP.

- Wiecki - [An Intuitive Guide to Bayesian Statistics](https://twiecki.io/pages/an-intuitive-guide-to-bayesian-statistics.html): upcoming course on Bayesian models with Python, from another of the authors of PyMC.

- Coursera courses:

  - [Bayesian methods in machine learning](https://www.coursera.org/learn/bayesian-methods-in-machine-learning): quite theoretical, but coding parts are present as well (with PyMC). Good if you're interested in gaining a good foundation and if maths (with some demonstrations at the blackboard as well!) doesn't scare you. Topics include: Bayesian methods in general, the Expectation-Maximization algorithm, variational inference, MCMC, Gaussian processes and Bayesian optimization.

  - [Probabilistic Deep Learning with TensorFlow 2](https://www.coursera.org/learn/probabilistic-deep-learning-with-tensorflow2): very good course on Bayesian statistics applied to deep learning, all done using TFP. Topics like probabilistic neural network layers, the Bayes by backprop algorithm and variational inference are covered.

- Coding "porting" of famous books: much in the style of the Davidson-Pilon's book, the PyMC team has coded up the examples of some famous books (among which Gelman's and McElreath's) and put everything [on GitHub](https://github.com/pymc-devs/resources).

- Other books (mentioned exclusively for completeness!) on statistical models and machine learning that also touch some of the topics explored here:

  - Bishop - [Pattern recognition and machine learning](https://link.springer.com/book/9780387310732).

  - McKay - [Information theory, inference and learning algorithms](https://books.google.it/books/about/Information_Theory_Inference_and_Learnin.html?id=AKuMj4PN_EMC&redir_esc=y)

  - Murphy - [Machine learning - A probabilistic perspective](https://probml.github.io/pml-book/book0.html): now part of a [series of three books](https://probml.github.io/pml-book/)

  - Hastie, Tibshirani, Firedman - [Element of statistical learning](https://hastie.su.domains/ElemStatLearn/): a classic on machine learning (two of the authors also contributed to a [another book](https://link.springer.com/book/10.1007/978-1-4614-7138-7) that presents topics from a higher level perspective and with less focus on the maths)

  - Goodfellow, Bengio, Courville - [Deep learning](https://www.deeplearningbook.org/): the bible of deep learning from the world leading experts.

  - Bishop, Bishop - [Deep Learning - Foundations and Concepts](https://www.bishopbook.com/): new book (2024) on deep learning by the author of one of the best books on machine learning in general.

**Suggestion:** as before, to avoid feeling overwhelmed start with one source and try to stick to it. The first entry has working code examples for literally everything that's discussed, is very hands-on and is fairly easy to follow - plus, TFP is among the libraries used.
