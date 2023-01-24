from models import LSTM_HP,LSTM_HP_ATT
import random
import operator
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchvision.transforms import Grayscale
from torchvision.transforms.functional import resize



class SegmentTree(object):

    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, \
            "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array.
        """
        if end is None:
            end = self._capacity - 1
        if end < 0:
            end += self._capacity
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, neutral_element=float('inf'))

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer:

    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, *args):
        if self._next_idx >= len(self._storage):
            self._storage.append(args)
        else:
            self._storage[self._next_idx] = args
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        b_o, b_a, b_r, b_o_, b_d = [], [], [], [], []
        for i in idxes:
            o, a, r, o_, d = self._storage[i]
            b_o.append(o)
            b_a.append(a)
            b_r.append(r)
            b_o_.append(o_)
            b_d.append(d)

        return (b_o,b_a,b_r,b_o_,b_d)

    def sample(self, batch_size):
        indexes = range(len(self._storage))
        idxes = [random.choice(indexes) for _ in range(batch_size)]
        return self._encode_sample(idxes)



class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, size, alpha, beta):

        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self.beta = beta

    def add(self, *args):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args)
        self._it_sum[idx] = self._max_priority**self._alpha
        self._it_min[idx] = self._max_priority**self._alpha

    def _sample_proportional(self, batch_size):
        '''
        Samplea proporcionalmente en segmentos sobre el storage.
        '''
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        idxes = self._sample_proportional(batch_size)

        it_sum = self._it_sum.sum()
        p_min = self._it_min.min() / it_sum
        max_weight = (p_min * len(self._storage))**(-self.beta)

        p_samples = np.asarray([self._it_sum[idx] for idx in idxes]) / it_sum
        weights = (p_samples * len(self._storage))**(-self.beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample + (weights.astype('float32'), idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions"""
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority**self._alpha
            self._it_min[idx] = priority**self._alpha

            self._max_priority = max(self._max_priority, priority)



env_model1= LSTM_HP(
    input_size=9,
    hidden_size=256,
    num_layers=2,
    fc_units=512,
    num_actions=5
)

 
env_model2= LSTM_HP_ATT(
    input_size=9,
    hidden_size=256,
    fc_units=512,
    num_actions=5
)


class DDQN:
    def __init__(self,output_shape,device,sync_steps):
        self.output_shape = output_shape
        self.device = device
        self.Q = env_model2.to(device)
        self.Q_target = env_model2.to(device)
        self.opt = Adam(self.Q.parameters(),0.00025)
        self.sync_steps = sync_steps
        self.iter = 1


    def preprocess_sample(self,*samples):
        b_o,b_a,b_r,b_o_,b_d,b_w= samples
        o = torch.cat(b_o).to(self.device).type(torch.float)
        a = torch.tensor(b_a).to(self.device).type(torch.int64) 
        r = torch.tensor(b_r).to(self.device).type(torch.float)
        d = torch.tensor(b_d).to(self.device).type(torch.float)
        o_ = torch.cat(b_o_).to(self.device).type(torch.float)
        w = torch.tensor(b_w).to(self.device).type(torch.float)
        return (o,a,r,d,o_,w)

    def error_func(self,*sample,gamma):
        s,a,r,d,s_,w = sample
        a_ = self.Q(s_).argmax(dim=1)
        q_value =torch.sum(F.one_hot(a_,self.output_shape).view(-1,self.output_shape)*self.Q_target(s_),dim=1)
        q_ = r+gamma*((1-d)*(q_value))
        q = torch.sum(F.one_hot(a,self.output_shape).view(-1,self.output_shape)*self.Q(s),dim=1)
        td_error = q-q_
        loss = torch.mean(torch.nn.HuberLoss(reduction='none')(q,q_)*w)
        return td_error,loss

    def train(self,b_o, b_a, b_r, b_o_, b_d, b_w,gamma):
        sample = self.preprocess_sample(b_o,b_a,b_r,b_o_,b_d,b_w)
        td_error,loss = self.error_func(*sample,gamma=gamma)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        if self.iter%self.sync_steps:
            self.iter = 1
            self.sync()
        else: 
            self.iter += 1            
        return td_error.to('cpu').detach().numpy()

    def sync(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def take_action(self,o,epsilon):

        if random.random()<epsilon:
            return int(random.random()*self.output_shape)
        else:
            action = self.Q(o).argmax(dim=1).cpu().detach().numpy()
            return int(action)    

           
    def preprocess_state(self,state):
        shape = state.shape
        return torch.tensor(state).view(1,shape[0],shape[1]).to(self.device).type(torch.float)
