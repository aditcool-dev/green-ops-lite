def grade(env):
    temps = env.rack_temp
    loads = env.cpu_load

    stability = sum(max(0, 1 - (t - 60)/30) for t in temps) / 3
    efficiency = max(0, 1 - (env.power_cost / 3))
    
    # Throughput metric. The agent MUST keep the servers processing jobs!
    throughput = sum(loads) / 3.0 

    # 40% Heat, 30% Power, 30% Server Uptime
    raw_score = 0.4 * stability + 0.3 * efficiency + 0.3 * throughput

    score = max(0.01, min(0.99, raw_score))

    return round(score, 4)