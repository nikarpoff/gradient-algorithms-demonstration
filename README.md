# gradient-algorithms-demonstration
Realisation and graphic demonstration of GD (Gradient Descent), CGD (Conjugate Gradient Descent), ADAM (Adaptive Moment Estimation), SQN (stochastic quasi­Newton), BFGS (Broyden–Fletcher–Goldfarb–Shanno)

You can find theory and practice in the `report.pdf` file.

# How to use
To start, you should have Python 3.

Clone repository:
```bash
git clone https://github.com/nikarpoff/gradient-algorithms-demonstration.git
```

Change directory:
```bash
cd gradient-algorithms-demonstration/
```

Install all dependencies:
```bash
pip install matplotlib numpy scipy
```

Run test example:
```bash
python main.py
```

You can change main.py to use another target function or gradient optimizer (just uncomment the required lines). You can also uncomment line `main.draw_solutions()` to unlock the graphic demonstration. Furthermore, you can configure hyperparameters for gradient optimizers.