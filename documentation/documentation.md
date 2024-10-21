# Magnetic Flux Rope Models (MFRM) documentation

The __Magnetic Flux Rope Models__ is a `Python` package whose aim is to provide a simple and reliable way to work with analytical magnetic flux rope models. It can be used to simply
explore and visualise the models, and also as artifical input for other processes such as the Grad-Shafranov reconstruction technique, or machine learning models.

__Details__:
- 👨🏽‍💻 __Author & developer__: [Jordi Jumilla Lorenz](https://github.com/jordijumilla/jordijumilla).
- 🎓 __Associations__: this code has been developed independently of any university or institution. However, it can be thought as a continuation of the author's thesis: [Connecting the Grad-Shafranov reconstruction technique to flux-rope models](https://upcommons.upc.edu/handle/2117/370244), with association with Universitat Politècnica de Catalunya (UPC) and NASA Goddard Space Flight Center.
- 🪪 __DOI__:


## Interface

This library contains an interface named *__InterfaceMFRModel__* which is an abstract class from which all magnetic flux models should be derived.

Methods that __must__ be implemented:
- `__init__`: standard method for instance initialisation.
- `_validate_params`: method to validate the user-defined parameters. This adds a layer of data validity.
- `simulate_crossing`: method to simulate a spacecraft crossing across the magnetic flux rope.

This interface also provides a __noise generator__ method called `get_noise_generator`, which can be used to generate reproducible pseudo-random samples. This is especially useful when trying to simulate the measurement errors from spacecraft.

Noise types currently supported:
- Uniform
- Gaussian

## Elliptical-cylindrical (EC) model

Implementation of the Nieves-Chinchilla et al. elliptical-cylindrical (EC) model defined in the article cited below.

It is worth noticing that the library uses a different convention for the elliptical coordinates. Instead of assuming that the rope axis lays on the y-axis and using  the coordinates $(r, y, \phi)$, the rope axis lays on the z-axis, and the coordinates used are $(r, \phi, z)$.

- 🗒️ __Article__: [Elliptic-cylindrical Analytical Flux Rope Model for Magnetic Clouds (2018)](https://doi.org/10.3847/1538-4357/aac951) - Teresa Nieves-Chinchilla et al.

## Circular-cylindrical (CC) model

The model was presented in 2016 by Nieves-Chinchilla et al., and can be seen as a particular case of the ECModel setting the delta parameter, which represents the ellipse distortion, to 1.

- 🗒️ __Article__: [A circular-cylindrical flux-rope analytical model for magnetic clouds (2016)](http://doi.org/10.3847/0004-637X/823/1/27) - Teresa Nieves-Chinchilla et al.

## Lundquist model

The Lundquist model is a well known analytical circular-cylindrical symmetric flux-rope model. It is a force-free model, obtained by setting $\mu_0 J = \alpha B$, with \alpha constant.

The magnetic field components are:

$B_r(r) = 0$

$B_{\phi}(r) = B_z^0 \cdot J_{1} \left( \alpha \cdot \dfrac{r}{R} \right)$

$B_z(r) = B_z^0 \cdot J_{0} \left( \alpha \cdot \dfrac{r}{R} \right)$

where $J_0$ and $J_1$ are the Bessel functions of first kind and zero-th and first order, respectively.
