# %% [markdown]
# Physics Simulation Using VPython
#
# From https://levelup.gitconnected.com/physics-simulations-using-vpython-a3d6ee69d121
#

# %%
import vpython as vp
import time
import itertools

print(vp.__version__)


vp.scene.title = "Modeling the motion of planets subject to gravitational forces"
vp.scene.height = 600
vp.scene.width = 800


def eval_gravitational_force(p1: vp.sphere, p2: vp.sphere):
    G = 1
    r = p1.pos - p2.pos
    distance = vp.mag(r)
    rhat = r / distance
    force = -rhat * G * p1.mass * p2.mass / distance ** 2
    return force


star = vp.sphere(
    pos=vp.vector(0, 0, 0),
    radius=0.2,
    color=vp.color.yellow,
    mass=2.0 * 1000,
    momentum=vp.vector(0, 0, 0),
    make_trail=True,
)

planets = [
    vp.sphere(
        pos=vp.vector(1, 0, 0),
        radius=0.05,
        color=vp.color.green,
        mass=1,
        momentum=vp.vector(0, 30, 0),
        make_trail=True,
    ),
    vp.sphere(
        pos=vp.vector(0, 3, 0),
        radius=0.075,
        color=vp.vector(0, 0.8, 0.3),
        mass=2,
        momentum=vp.vector(-35, 0, 0),
        make_trail=True,
    ),
    vp.sphere(
        pos=vp.vector(0, -4, 0),
        radius=0.1,
        color=vp.vector(0.6, 0.15, 0.68),
        mass=10,
        momentum=vp.vector(160, 0, 0),
        make_trail=True,
    ),
    # a small moon?
    vp.sphere(
        pos=vp.vector(0, -4.2, 0),
        radius=0.001,
        color=vp.vector(0.6, 0.15, 0.68),
        mass=0.1,
        momentum=vp.vector(1, 0, 0),
        make_trail=True,
    ),
]


bodies = planets + [
    star,
]


# %%
t = 0
dt = 0.001


while True:
    vp.rate(500)

    for b in bodies:
        b.force = vp.vector(0, 0, 0)

    for b1, b2 in itertools.product(bodies, bodies):
        if b1 != b2:
            b1.force += eval_gravitational_force(b1, b2)

    for b in bodies:
        b.momentum += b.force * dt
        b.pos += b.momentum / b.mass * dt

    t += dt
    time.sleep(0.1)
