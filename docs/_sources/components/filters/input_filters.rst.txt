Input Filters
=============

The input filters are separated into two categories - **observation filters** and **reward filters**.

Observation Filters
-------------------

ObservationClippingFilter
+++++++++++++++++++++++++
.. autoclass:: rl_coach.filters.observation.ObservationClippingFilter

ObservationCropFilter
+++++++++++++++++++++
.. autoclass:: rl_coach.filters.observation.ObservationCropFilter

ObservationMoveAxisFilter
+++++++++++++++++++++++++
.. autoclass:: rl_coach.filters.observation.ObservationMoveAxisFilter

ObservationNormalizationFilter
++++++++++++++++++++++++++++++
.. autoclass:: rl_coach.filters.observation.ObservationNormalizationFilter

ObservationReductionBySubPartsNameFilter
++++++++++++++++++++++++++++++++++++++++
.. autoclass:: rl_coach.filters.observation.ObservationReductionBySubPartsNameFilter

ObservationRescaleSizeByFactorFilter
++++++++++++++++++++++++++++++++++++
.. autoclass:: rl_coach.filters.observation.ObservationRescaleSizeByFactorFilter

ObservationRescaleToSizeFilter
++++++++++++++++++++++++++++++
.. autoclass:: rl_coach.filters.observation.ObservationRescaleToSizeFilter

ObservationRGBToYFilter
+++++++++++++++++++++++
.. autoclass:: rl_coach.filters.observation.ObservationRGBToYFilter

ObservationSqueezeFilter
++++++++++++++++++++++++
.. autoclass:: rl_coach.filters.observation.ObservationSqueezeFilter

ObservationStackingFilter
+++++++++++++++++++++++++
.. autoclass:: rl_coach.filters.observation.ObservationStackingFilter

ObservationToUInt8Filter
++++++++++++++++++++++++
.. autoclass:: rl_coach.filters.observation.ObservationToUInt8Filter


Reward Filters
--------------

RewardClippingFilter
++++++++++++++++++++
.. autoclass:: rl_coach.filters.reward.RewardClippingFilter

RewardNormalizationFilter
+++++++++++++++++++++++++
.. autoclass:: rl_coach.filters.reward.RewardNormalizationFilter

RewardRescaleFilter
+++++++++++++++++++
.. autoclass:: rl_coach.filters.reward.RewardRescaleFilter
