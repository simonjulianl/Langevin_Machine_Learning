import torch
from HNNwLJ20210507.src20210507.test.unused.fields_interpolate.phase_space import phase_space


boxsize = 4

q0x = boxsize*3e9+0.1
q0y = -1*boxsize*2+2.5
q1x = 3.2
q1y = boxsize*7+1.4


q_list = [[[q0x  ,q0y ],[q1x ,q1y ]]]
q_list = torch.tensor(q_list)

print('before adjust', q_list )

_phase_space = phase_space()
_phase_space.adjust_real(q_list, boxsize)

print('after adjust', q_list )

