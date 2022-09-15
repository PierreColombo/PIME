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
>>> import pime
>>> import torch
>>> N = 5
>>> KL_div = pime.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pime.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pime.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5
>>> KL_div = pime.example.KL()
>>> reference_distribution = torch.tensor([1/N]*N)
>>> input_distribution = torch.random(N)
>>> KL_div.predict(reference_distribution,input_distribution)
>>> import torch
>>> N = 5


