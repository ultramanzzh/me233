## ME233
Enhanced CANN for Different Hyperelastic Materials Model Discovery: Adaptive Log-Exp Parameterization with Cubic Terms

To install all dependencies, run:
````bash
pip install -r requirements.txt
````

To train and test the model, run:
````bash
python train.py  # train with tension by default, "--t" -tension; "--c" - compression; "--shear" -shear
python test.py  # test the tension by default, "--t" -tension; "--c" - compression; "--shear" -shear
````

Make use of *L2-regularzation* and *Early Stop* to prevent overfit.
The dataset contained comes from ME233 class material @Stanford

03/08/2025 Note that an error will be raised when learning epochs are smaller than 1000, due to defects in algorithm Early Stop.
Also, note than when there are too many data points or the stretch is getting larger than some thresholds, some weights may get to infinity.
The specific reason is still under-discovered. The threshold is not known yet.
I'll fix that if have a chance. My idea is use some initialization towards the weights, encouraging exp term to grow.

