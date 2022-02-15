# ImageRegression
![anim.gif](.assets\anim.gif)

Preliminary studies during my MSCS thesis on boosting Neural Radiance Fields. This served as pratice with PyTorch by implementing and debugging a toy problem that is a coordinate-based image regression model.

## Usage
Run one of the 3 main runnable scripts: `src/reg_base.py`, `src/reg_grownet.py`, and `src/reg_xgboost.py`
- (base script uses a single MLP, the grownet one relies on an ensemble of smaller MLPs, and the last one is an implementation in xgboost)
- XGBoost note: i don't know if it is possible to learn correlated multi-dimensional outputs, so I simply fit each channel separately 
- for some command line examples, see `runner.ps1`

See `config/default.py` for the default configurations.
There are already some images of mine in `data/`.

## Development Chronology
- learning PyTorch boilerplate and Autograd inner workings for gradient computation and backpropagation
- started with a simple ReLU model → blurry outputs
- added a positional encoding to the ReLU model to counteract the MLP bias towards low frequencies  (see https://bmild.github.io/fourfeat)
- found about a paper which encodes the high frequency details directly in the MLP activation functions (https://github.com/vsitzmann/siren), so I added a SIREN model as alternative
- I wanted to collect results, so I played around with Weights&Biases. It has some features like dataset versioning, and model management, but to me it was very handy for the possibility to see the experiments real-time from any device 
    - some of the experiments are available here (a bit messy though): https://wandb.ai/a-di/image_regression_final/workspace?workspace=user-a-di
- running experiments purely from config files was not scalable, so I added the parameters as command line arguments

## Credits
- `grownet.py` is adapted from https://github.com/sbadirli/GrowNet
-  SIREN models in `models.py` are adapted from https://github.com/vsitzmann/siren


