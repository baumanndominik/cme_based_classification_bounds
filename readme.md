# Frequentist bounds for multi-class classification 

This code accompanies the the paper "Frequentist bounds for multi-class classification with applications to safe reinforcement learning."

## Requirements

Code was developed using Python 3.8.10. The following libraries are required:

* numpy (developed with version 1.21.0)
* scikit-learn (developed with version 1.0.2)
* torchvision (developed with version 0.7.0)
* matplotlib (developed with version 3.3.4)

## Execution

To execute the code, run

```
python main.py
```

By default, the code runs for 10'000 iterations on the OpenAI Gym [1] continuous mountain car environment. At each iteration, a context is sampled and a measurement obtained. The learning agent tries to infer the context from the measurement. If it is too uncertain, it queries an oracle instead and improves its estimates. The overall goal is to learn a policy for the mountain car, which is achieved through a safe learning algorithm [2]. To control the runtime of the algorithm, we terminate policy optimization after 1'000 iterations.

## References

* [1] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba, "OpenAI gym," arXiv preprint arXiv:1606.01540, 2016.
* [2] Felix Berkenkamp, Andreas Krause, and Angela P Schoellig, "Bayesian optimization with safety constraints: safe and automatic parameter tuning in robotics," Machine Learning, pages 1–35, 2021.