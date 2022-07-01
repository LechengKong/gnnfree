import torch

class BinaryLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, scores, labels):
        if labels.dim()<=1:
            labels = labels.unsqueeze(1)
        is_labeled = torch.logical_not(torch.isnan(labels))
        loss = 0
        target_res = scores[is_labeled]
        target_label = labels[is_labeled].to(torch.float)
        loss = self.loss(target_res, target_label)
        return loss

class MultiClassLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, scores, labels):
        loss = self.loss(scores, labels)
        return loss

class InfoNCEloss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, MI_MAT):
        n = len(MI_MAT)
        e_neg_mat = MI_MAT.view(-1)[1:].view(n-1,n+1)[:,:-1].reshape(n,n-1)
        e_pos = torch.diagonal(MI_MAT)
        loss = -torch.mean(torch.log(torch.exp(e_pos)/torch.exp(e_neg_mat).sum(dim=-1)))
        return loss


class CCALoss(torch.nn.Module):
    def __init__(self, outdim_size=20):
        super().__init__()
        self.outdim_size = outdim_size

    def forward(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """

        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-7

        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)
#         print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=H1.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=H1.device)
        # print(SigmaHat11.sum())
        # print(SigmaHat22.sum())
        # print(SigmaHat12.sum())
        assert torch.isnan(SigmaHat11).sum().item() == 0
        assert torch.isnan(SigmaHat12).sum().item() == 0
        assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.linalg.eigh(SigmaHat11)
        [D2, V2] = torch.linalg.eigh(SigmaHat22)
        # v,c = torch.unique(D1, return_counts=True)
        # print(v[c>1])
        # v,c = torch.unique(D2, return_counts=True)
        # print(v[c>1])
        assert torch.isnan(D1).sum().item() == 0
        assert torch.isnan(D2).sum().item() == 0
        assert torch.isnan(V1).sum().item() == 0
        assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)                                
#         print(Tval.size())

        # just the top self.outdim_size singular values are used
        trace_TT = torch.matmul(Tval.t(), Tval)
        trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(H1.device)) # regularization for more stability
        U, V = torch.linalg.eigh(trace_TT)
        U = torch.where(U>eps, U, (torch.ones(U.shape)*eps).to(H1.device))
        U = U.topk(self.outdim_size)[0]
        corr = torch.sum(torch.sqrt(U))
        U, S, V = torch.svd(Tval)
        U = U[:,:self.outdim_size]
        V = V[:,:self.outdim_size]
        U = torch.matmul(SigmaHat11RootInv, U)
        V = torch.matmul(SigmaHat22RootInv, V)
        return corr, U, V

class IDLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, res):
        return res

class MSELoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.MSELoss()
    
    def forward(self, pred, target):
        return self.loss(pred, target)