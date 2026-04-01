def grade(env):
    temps = env.rack_temp

    stability = sum(1 for t in temps if t < 70) / 3
    uptime = 1.0 if all(t < 85 for t in temps) else 0.0

    return round(0.7 * stability + 0.3 * uptime, 3)