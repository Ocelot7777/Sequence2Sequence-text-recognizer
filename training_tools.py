import torch


class TrainingTools(object):
    ''' some static methods to help with model training'''

    @staticmethod
    def data_to_device(data, device):
        '''send data(s) to device'''
        if isinstance(data, dict):
            return {k: TrainingTools.data_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [TrainingTools.data_to_device(sub_data, device) for sub_data in data]
        else:
            return data.to(device) if isinstance(data, torch.Tensor) else data

    @staticmethod
    def clip_grad(net, max_grad=10.):
        """Computes a gradient clipping coefficient based on gradient norm."""
        total_norm = 0
        for p in net.parameters():
            if p.requires_grad:
                modulenorm = p.grad.data.norm()
                total_norm += modulenorm ** 2

        total_norm = math.sqrt(total_norm)

        norm = max_grad / max(total_norm, max_grad)
        for p in net.parameters():
            if p.requires_grad:
                p.grad.mul_(norm)