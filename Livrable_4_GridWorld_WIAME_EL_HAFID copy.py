import numpy as np
import pygame
import sys

# ---------------- GridWorld with Obstacle ---------------- #
class GridWorld:
    def __init__(self, n=5, obstacle=(2,2)):
        self.n = n
        self.start = (0, 0)
        self.goal = (n-1, n-1)
        self.obstacle = obstacle
        self.state = self.start
    
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 0 and x > 0:          # up
            x -= 1
        elif action == 1 and x < self.n-1: # down
            x += 1
        elif action == 2 and y > 0:        # left
            y -= 1
        elif action == 3 and y < self.n-1: # right
            y += 1

        # If move goes into obstacle, stay in same place
        if (x, y) == self.obstacle:
            x, y = self.state

        self.state = (x, y)

        # Reward + done
        if self.state == self.goal:
            return self.state, 1, True, {}
        else:
            return self.state, 0, False, {}

# ---------------- Policy ---------------- #
def random_policy(state):
    return np.random.choice(4)  # up, down, left, right

# ---------------- Features ---------------- #
def phi(state, n=5):
    vec = np.zeros(n*n)
    idx = state[0]*n + state[1]
    vec[idx] = 1.0
    return vec

# ---------------- TD(0) Agent ---------------- #
class TD0Agent:
    def __init__(self, n, alpha=0.1, gamma=0.99):
        self.n = n
        self.d = n*n
        self.w = np.zeros(self.d)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, s, r, s2, done):
        phi_s = phi(s, self.n)
        phi_s2 = phi(s2, self.n)
        delta = r + (0 if done else self.gamma * np.dot(self.w, phi_s2)) - np.dot(self.w, phi_s)
        self.w += self.alpha * delta * phi_s

    def value_grid(self):
        return self.w.reshape((self.n, self.n))

# ---------------- Visualization with pygame ---------------- #
def run_game(n=5, obstacle=(2,2), episodes=500):
    # init env + agent
    env = GridWorld(n=n, obstacle=obstacle)
    agent = TD0Agent(n=n, alpha=0.1, gamma=0.99)

    # pygame setup
    pygame.init()
    cell_size = 80
    screen = pygame.display.set_mode((n*cell_size, n*cell_size))
    pygame.display.set_caption("TD(0) GridWorld Learning")

    clock = pygame.time.Clock()
    trajectory = []  # store visited states

    for ep in range(episodes):
        s, done = env.reset(), False
        trajectory.clear()  # reset trace each episode
        while not done:
            # Handle quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Choose action + step
            a = random_policy(s)
            s2, r, done, _ = env.step(a)

            # Update TD(0)
            agent.update(s, r, s2, done)

            # Save trajectory
            trajectory.append(env.state)

            # --- DRAW GRID --- #
            screen.fill((0,0,0))
            values = agent.value_grid()
            vmax = np.max(values) if np.max(values) > 0 else 1.0

            for i in range(n):
                for j in range(n):
                    rect = pygame.Rect(j*cell_size, i*cell_size, cell_size, cell_size)
                    # obstacle
                    if (i,j) == obstacle:
                        color = (200,0,0)
                    # goal
                    elif (i,j) == env.goal:
                        color = (0,200,0)
                    else:
                        # map value estimate to blue intensity
                        v = values[i,j] / vmax
                        v = max(0.0, min(1.0, v))
                        color = (int(50), int(50 + 200*v), int(50))
                    pygame.draw.rect(screen, color, rect)
                    pygame.draw.rect(screen, (100,100,100), rect, 2)

            # draw trajectory (faded white dots)
            for t, (x,y) in enumerate(trajectory):
                alpha = max(50, 255 - (len(trajectory)-t)*20)  # fading
                alpha = min(alpha, 255)
                trace_color = (alpha, alpha, alpha)
                pygame.draw.circle(screen, trace_color,
                                   (y*cell_size+cell_size//2, x*cell_size+cell_size//2),
                                   8)

            # agent (blue circle)
            ax, ay = env.state
            pygame.draw.circle(screen, (0,0,255),
                               (ay*cell_size+cell_size//2, ax*cell_size+cell_size//2),
                               cell_size//3)

            pygame.display.flip()
            clock.tick(5)  # speed (FPS)

            s = s2

    pygame.quit()

# ---------------- Run ---------------- #
if __name__ == "__main__":
    run_game(n=5, obstacle=(2,2), episodes=1000)
