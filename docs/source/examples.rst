Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pim


Computing MI measure between two continuous R.V
----------------
>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Computing the differential Entropy of a R.V
----------------
>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Computing the MI as a difference of two Entropy
----------------
>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Computing discrepency measure between two discret distributions
----------------

Let's see how can we use PIM to compute a discrepancy measure between two distributions.

You can use for example the KL divergence ``pimms.example.KL``:

.. autofunction::pimms.example.KL

For example:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import lumache
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

>>> import pim
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)


Many more discrepancy measures are available such as:

Computing MI measure between a continuous and a discret R.V
----------------
>>> import pimms
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pimms.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)

