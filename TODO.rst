====
TODO
====

Spike detection
===============

Threshold detection
-------------------
    * Online estimate of the median voltage value for each electrode
    * Online estimate of the median voltage absolute deviation for each electrode
    * Detect peaks (threshold crosssings)
    * Use a ring buffer (2x input buffer size)?

Feature extraction
------------------
    * Collect detected peaks (until M peaks have been collected?)
    * Compute the PCA for peak waveforms
    * Compute the peak features vectors
      (i.e. PCA projections for each electrodes for a given peak time)
    * Compute the PCA for peak features vectors
    * Compute the spike features vector
      (i.e. PCA projection for a given peak time)

Clustering
----------
    * Cluster the spike features vector in real-time
