Benchmarks
==========

Reinforcement learning is a developing field, and so far it has been particularly difficult to reproduce some of the
results published in the original papers. Some reasons for this are:

* Reinforcement learning algorithms are notoriously known as having an unstable learning process.
  The data the neural networks trains on is dynamic, and depends on the random seed defined for the environment.

* Reinforcement learning algorithms have many moving parts. For some environments and agents, there are many
  "tricks" which are needed to get the exact behavior the paper authors had seen. Also, there are **a lot** of
  hyper-parameters to set.

In order for a reinforcement learning implementation to be useful for research or for data science, it must be
shown that it achieves the expected behavior. For this reason, we collected a set of benchmark results from most
of the algorithms implemented in Coach. The algorithms were tested on a subset of the same environments that were
used in the original papers, and with multiple seed for each environment.
Additionally, Coach uses some strict testing mechanisms to try and make sure the results we show for these
benchmarks stay intact as Coach continues to develop.

To see the benchmark results, please visit the
`following GitHub page <https://github.com/NervanaSystems/coach/tree/master/benchmarks>`_.