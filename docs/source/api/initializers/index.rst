.. _api.initializers:

#########################
pysatl_mpest.initializers
#########################

.. automodule:: pysatl_mpest.initializers
   :no-members:

------------------

Initializers
============

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~ClusterInitializer
   ~MultiStartInitializer
   ~RefiningInitializer

Abstract Classes
================

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~Initializer
   ~Partitioner
   ~ComponentEstimator
   ~Matcher
   ~ComponentTable

Partitioners
============

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~KMeansPartitioner
   ~QuantilePartitioner
   ~RandomCentersPartitioner
   ~SklearnPartitioner

Component Estimators
====================

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~MLEComponentEstimator

Matchers
========

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~HungarianMatcher
   ~IdentityMatcher

Exceptions
==========

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~InitializationError
   ~ComponentEstimationError
