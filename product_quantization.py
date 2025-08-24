# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 13:54:06 2025

@author: Jose Antonio
"""

from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted
import numpy as np

class ProductQuantizer:
    def __init__(
            self,
            d_emb: int,
            m: int,
            n_bits: int
            ) -> None:
        
        self.d_emb = d_emb
        self.m = m
        self.n_bits = n_bits
        self.k = 2**n_bits
        self.d_subemb = d_emb // m
        self.estimators = [KMeans(n_clusters=self.k) for _ in range(m)]
        self.trained = False
    
    
    def train(self, embeddings: np.ndarray) -> None:
        """
        Trains all the estimators on the corpus of embeddings

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n, d_emb) with n being the number of embeddings.

        Returns
        -------
        None
            Trains all the estimators.

        """
        
        for i, estimator in enumerate(self.estimators):
            print(f"Fitting estimator {i+1}")
            splited_embeddings = embeddings[:, i * self.d_subemb : (i+1)*self.d_subemb]
            estimator.fit(splited_embeddings)
            self.trained = True
    
    
    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """
        Encodes an array of embeddings

        Parameters
        ----------
        embedding : np.ndarray
            Shape [n, m] where m is the number of subvectors and n the number of embeddings to quantize.

        Raises
        ------
        Exception
            In case the method is called without calling train method.

        Returns
        -------
        None.

        """
        if not self.trained:
            raise Exception("Train before encoding")
        
        result = np.empty((embedding.shape[0], self.m))
        
        for i, estimator in enumerate(self.estimators):
            splitted_embedding = embedding[:, i*self.d_subemb:(i+1)*self.d_subemb]
            result[:, i] = estimator.predict(splitted_embedding)
        
        return result
    
            
if __name__ == "__main__":
    pq = ProductQuantizer(d_emb=10, m=2, n_bits=2)
    pq.train(np.random.uniform(10, 100, size=(10,10)).astype(np.float32))
            
        
    
        
        
        
        