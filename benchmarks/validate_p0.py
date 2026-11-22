import math, torch
from lawless.physics.constants import PhysicsConfig
from lawless.physics.state import make_state
from lawless.physics.simulator import step


cfg=PhysicsConfig(); steps=round(40/cfg.dt)
torch.manual_seed(42)
s=make_state((4,256)); s.velocity.uniform_(-1,1)
s.angle.uniform_(-3,3); s.angular_velocity.uniform_(-1,1)
s.energy.uniform_(0,100)
a=torch.rand(4,256,2)*2-1; g=torch.randn(4,1,2)
fields=("position","velocity","angle","angular_velocity","energy")
def check(x):
    for f in fields:
        assert torch.isfinite(getattr(x,f)).all(), f
    assert (x.energy>=0).all() and (x.energy<=100).all()
    assert (x.angle>=-math.pi).all() and (x.angle<math.pi).all()
def run(dev):
    x=s.to(dev); ac=a.to(dev); gr=g.to(dev)
    for _ in range(steps):
        x,_=step(x,ac,gr,cfg); check(x)
    return x
cpu=run("cpu")

if torch.cuda.is_available():
    gpu=run("cuda")
    for f in fields:
        diff=(getattr(cpu,f)-getattr(gpu,f).cpu()).abs().max().item()
        print(f,"max_abs_deviation:",diff)
    assert torch.equal(cpu.alive,gpu.alive.cpu())
print("P0 validation:",steps,"steps,",steps*cfg.dt,"seconds")
