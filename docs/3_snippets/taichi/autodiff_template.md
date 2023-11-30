---
sidebar_position: 2
---

# Taichi differentiable template combined with Pytorch
refer to [3D-Tools](https://github.com/DaLi-Jack/3D-Tools/blob/main/taichi/AutoDiff.py)
```python
import torch
import torch.nn as nn
import taichi as ti
import trimesh

# init taichi
arch = ti.cuda
ti.init(arch=arch)


@ti.kernel
def calculate(
    n_particles:int,
    pc:ti.template(),
    pc_after:ti.template(),
):
    
    for i in range(n_particles):
        pc_after[i][0] = pc[i][0] * pc[i][0]
        pc_after[i][1] = ti.exp(pc[i][1])
        pc_after[i][2] = ti.sin(pc[i][2])
        

class Template(torch.nn.Module):
    def __init__(self):
        super(Template, self).__init__()

        self.n_particles = 100
        self.device = 'cuda'

        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, obj_pc):       # obj_pc: [n, 3], torch.tensor (cuda)

                ctx.input_size = obj_pc.shape
                self.n_particles = obj_pc.shape[0]
                self.device = obj_pc.device

                # define the output tensor
                output_pc = torch.zeros_like(obj_pc, device=self.device, requires_grad=True)

                # convert torch.tensor to ti.field
                self.obj_pc_before = ti.Vector.field(3, float, self.n_particles, needs_grad=True)
                self.obj_pc_before.from_torch(obj_pc.contiguous())
                self.obj_pc_after = ti.Vector.field(3, float, self.n_particles, needs_grad=True)

                # run the taichi kernel
                calculate(self.n_particles, self.obj_pc_before, self.obj_pc_after)

                # convert ti.field to torch.tensor
                output_pc = self.obj_pc_after.to_torch(device=self.device)

                return output_pc
            

            @staticmethod
            def backward(ctx, dL_dpx_after):

                input_size = ctx.input_size

                # define the input grad tensor
                input_grad = torch.zeros(*input_size, dtype=dL_dpx_after.dtype, device=self.device)

                # assign the output grad from torch to ti.field
                self.obj_pc_after.grad.from_torch(dL_dpx_after.contiguous())

                # back propagate gard along the taichi kernel
                calculate.grad(self.n_particles, self.obj_pc_before, self.obj_pc_after)

                # assign the input grad from ti.field to torch
                input_grad = self.obj_pc_before.grad.to_torch(device=self.device)

                return input_grad
        
        self._module_function = _module_function.apply
        
    def forward(self, obj_pc):

        return self._module_function(obj_pc)


if __name__ == "__main__":

    obj_pc = trimesh.load('./data/pc.obj').vertices
    obj_pc = torch.tensor(obj_pc, dtype=torch.float32, device='cuda', requires_grad=True)

    template = Template()
    L1_loss = nn.L1Loss(reduction='sum')

    obj_pc_after = template(obj_pc)
    loss = L1_loss(obj_pc_after, obj_pc)
    print('loss: ', loss)

    loss.backward()
    print('obj_pc.grad: ', obj_pc.grad)
    
    print('done')
```