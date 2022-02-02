# hybridEB
[![tests](https://github.com/beckermr/hybrideb/actions/workflows/tests.yml/badge.svg)](https://github.com/beckermr/hybrideb/actions/workflows/tests.yml)

hybridEB computes the hybrid real/fourier-space band-power E- and B-mode estimators for cosmic shear from Becker & Rozo (2015; http://arxiv.org/abs/1412.3851).

# example

```python
import hybridEB

theta_min = 1 # arcmin
theta_max = 400 # arcmin
Ntheta = 1000 # number of bins in log(theta)

heb = hybrideb.HybridEB(theta_min, theta_max, Ntheta)
beb = hybrideb.BinEB(theta_min, theta_max, Ntheta)

geb = hybrideb.GaussEB(beb, heb)

res = geb(3)  # grab the third estimator
theta_rad = res[0]
# X+ = np.sum((fp*xip + fm*xim)/2)
# X- = np.sum((fp*xip - fm*xim)/2)
fp = res[1]
fm = res[2]

# X+ = \int ell factors(ell) (wp * Pe + wm * Pb)
# X- = \int ell factors(ell) (wm * Pe + wp * Pb)
ell = res[3]
wp = res[4]
wm = res[5]
```
