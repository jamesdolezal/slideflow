from __future__ import division

import numpy as np
from typing import Tuple

import slideflow.norm.utils as ut


class Normalizer(ut.BaseNormalizer):
    """
    A stain normalization object
    """

    def __init__(self):
        super().__init__()

        HERef = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        maxCRef = np.array([1.9705, 1.0308])
        self.stain_matrix_target = HERef
        self.target_concentrations = maxCRef
        self.autofit = False

    def fit(self, img: np.ndarray, Io=255, alpha=1, beta=0.15) -> None:
        """Fit to a target image."""
        HE, maxC, _ = self.matrix_and_concentrations(img, Io, alpha, beta)
        self.stain_matrix_target = HE
        self.target_concentrations = maxC

    @staticmethod
    def matrix_and_concentrations(
        img: np.ndarray,
        Io: int = 255,
        alpha: float = 1,
        beta: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # reshape image
        img = img.reshape((-1, 3))

        img = ut.standardize_brightness(img)

        # calculate optical density
        OD = -np.log((img.astype(float) + 1) / Io)

        # remove transparent pixels
        ODhat = OD[~np.any(OD < beta, axis=1)]

        # compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:, 1:3])

        phi = np.arctan2(That[:, 1],That[:, 0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100-alpha)

        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T

        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]

        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :],99)])

        return HE, maxC, C

    def transform(
        self,
        img: np.ndarray,
        Io: int = 255,
        alpha: float = 1,
        beta: float = 0.15
    ) -> np.ndarray:
        ''' Normalize staining appearence of H&E stained images

        Example use:
            see test.py

        Input:
            I: RGB input image
            Io: (optional) transmitted light intensity

        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image

        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''
        # define height and width of image
        h, w, c = img.shape

        HERef = self.stain_matrix_target
        maxCRef = self.target_concentrations

        HE, maxC, C = self.matrix_and_concentrations(img, Io, alpha, beta)

        tmp = np.divide(maxC, maxCRef)
        C2 = np.divide(C, tmp[:, np.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
        #Inorm[Inorm > 255] = 254
        Inorm = np.clip(Inorm, 0, 255)
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

        # unmix hematoxylin and eosin
        H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
        #H[H > 255] = 254
        H = np.clip(H, 0, 255)
        H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

        E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
        #E[E > 255] = 254
        E = np.clip(E, 0, 255)
        E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

        return Inorm
