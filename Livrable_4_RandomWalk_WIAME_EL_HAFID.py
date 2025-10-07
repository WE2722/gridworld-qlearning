import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------- Environment ---------------- #
class RandomWalkEnv:
    def __init__(self, n=5):
        self.n = n  # states 0..n
        self.reset()
    
    def reset(self):
        self.state = self.n // 2
        return self.state
    
    def step(self, action):
        if np.random.rand() < 0.5:
            self.state -= 1
        else:
            self.state += 1
        if self.state == 0:  # left terminal
            return self.state, 0, True, {}
        elif self.state == self.n:  # right terminal
            return self.state, 1, True, {}
        else:
            return self.state, 0, False, {}

# ---------------- Helpers ---------------- #
def pi(s):
    return None  # dummy policy

def phi(s, n=5):
    vec = np.zeros(n+1)
    vec[s] = 1.0
    return vec

def td0_linear_prediction_live(env, pi, phi, alpha=0.05, gamma=0.99, episodes=5000):
    d = len(phi(env.reset()))
    w = np.zeros(d)
    snapshots = []

    for episode in range(episodes):
        s, done = env.reset(), False
        while not done:
            a = pi(s)
            s2, r, done, _ = env.step(a)
            phi_s = phi(s)
            phi_s2 = phi(s2)
            delta = r + (0 if done else gamma * np.dot(w, phi_s2)) - np.dot(w, phi_s)
            w += alpha * delta * phi_s
            s = s2

        # Save snapshot every 20 episodes for animation
        if (episode+1) % 20 == 0:
            snapshots.append(w.copy())
    return w, snapshots

# ---------------- Run TD(0) ---------------- #
env = RandomWalkEnv(n=5)
final_w, snapshots = td0_linear_prediction_live(env, pi, phi, episodes=2000)

true_values = np.array([s/5 for s in range(6)])

# ---------------- Animation ---------------- #
fig, ax = plt.subplots(figsize=(8,5))
line, = ax.plot([], [], 'bo-', label="Estimated values")
true_line, = ax.plot(true_values, 'k--', label="True values")
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(0, 5)
ax.set_xlabel("State")
ax.set_ylabel("Value estimate")
ax.set_title("TD(0) Learning - Random Walk")
ax.legend()

def init():
    line.set_data([], [])
    return line,

def update(frame):
    w = snapshots[frame]
    values = [np.dot(w, phi(s)) for s in range(6)]
    line.set_data(range(6), values)
    ax.set_title(f"TD(0) Learning - Random Walk (Episode {(frame+1)*20})")
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(snapshots), init_func=init,
                              interval=200, blit=True, repeat=False)

plt.show()
