class AverageMeter(object):
    """Computes and stores the average and current value, copied from Pytorch Example"""
    def __init__(self, NVars=1, ):
        self.NVars=NVars
        self.reset()
    def reset(self, ):
        self.sum = [0 for i in range(self.NVars)]
        self.cnt = 0
    def update(self, val, n=1):
        for i in range(self.NVars):
            self.sum[i] += val[i] * n
        self.cnt += n
    def avg(self,):
        if self.cnt:
            return tuple([self.sum[i]/self.cnt for i in range(self.NVars)])
        else:
            warn('No data updated, zeros returned')
            return tuple([0]*self.NVars)
