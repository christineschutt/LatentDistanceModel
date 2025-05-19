import numpy as np

def NMI_advanced(Aij_probs_true, Aij_probs_pred):
        """
    Computes the Normalized Mutual Information (NMI) between two probit matrix stored in tensors between runs for two models.

    Parameters:
    -----------
    Aij_probs_true : torch.Tensor
        A 3D tensor of shape (C, N, M) representing the predicted probability distributions over `C` ordinal classes
        for each drug-side effect pair
    
    Aij_probs_pred : torch.Tensor
        A 3D tensor of shape (C, N, M) representing the predicted probability distributions over `C` ordinal classes
        for each drug-side effect pair

    Returns:
    --------
    NMI : float
        The Normalized Mutual Information between the two probit-matrices. This score quantifies the
        mutual dependence between the two distributions, normalized by their entropies.
    
    joint_probs : np.ndarray
        A 2D array of shape (C, C) representing the joint probability matrix between the class predictions
        from the two models.
        """
        Aij_probs_true = Aij_probs_true.detach().cpu().numpy()
        Aij_probs_pred = Aij_probs_pred.detach().cpu().numpy()
        n_ordinal_classes, n_drugs, n_effects = Aij_probs_true.shape

        joint_probs = np.zeros((n_ordinal_classes, n_ordinal_classes))

        for i in range(n_drugs):
            for j in range(n_effects):
                joint_probs += np.outer(Aij_probs_true[:, i, j], Aij_probs_pred[:, i, j])
                
        joint_probs /= (n_drugs*n_effects)

        joint_true = np.zeros((n_ordinal_classes, n_ordinal_classes))
        joint_pred = np.zeros((n_ordinal_classes, n_ordinal_classes))

        for i in range(n_drugs):
             for j in range(n_effects):
                  joint_true += np.outer(Aij_probs_true[:, i, j], Aij_probs_true[:, i, j])
                  joint_pred += np.outer(Aij_probs_pred[:, i, j], Aij_probs_pred[:, i, j])

        joint_true /= (n_drugs * n_effects)
        joint_pred /= (n_drugs * n_effects)

        p_c = np.sum(joint_probs, axis=1)       # shape (n_classes,)
        p_c_mark = np.sum(joint_probs, axis=0)  # shape (n_classes,)

        # Compute entropy
        H_c = -np.sum(p_c * np.log(p_c + 1e-8))
        H_c_mark = -np.sum(p_c_mark * np.log(p_c_mark + 1e-8))

        # Compute mutual information
        MI = 0
        for c in range(n_ordinal_classes):
            for c_mark in range(n_ordinal_classes):
                if joint_probs[c, c_mark] > 0:
                    MI += joint_probs[c, c_mark] * np.log(joint_probs[c, c_mark] / (p_c[c] * p_c_mark[c_mark]))

        # Compute NMI
        NMI = 2 * MI / (H_c + H_c_mark + 1e-8)

        return NMI, joint_probs