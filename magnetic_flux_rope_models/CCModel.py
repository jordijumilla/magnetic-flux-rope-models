from magnetic_flux_rope_models.ECModel import ECModel

class CCModel(ECModel):
    """The circular-cylindrical (CC) model is a simplification of the elliptical-cylindrical (EC) model, for which the ellipse is a
    circle, that is, delta = 1. The CC model was developed before the EC model, therefore one can also see the EC model as a generalisation
    of the CC model.
    In any case, from the coding point of view, the CC model class is identical to the EC model class, but with delta set to 1."""
    def __init__(
        self,
        R: float,
        n: int = 1,
        m: int = 0,
        C_nm: float = 1.0,
        tau: float = 1.3,
        B_z_0: float = 10.0,
        handedness: int = 1
    ):
        super().__init__(
            delta=1.0,
            psi=0.0,
            R=R,
            n=n,
            m=m,
            C_nm=C_nm,
            tau=tau,
            B_z_0=B_z_0,
            handedness=handedness
        )
