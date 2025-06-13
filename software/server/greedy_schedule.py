import numpy as np

# === Greedy scheduling for deferrable demand ===
def greedy_schedule(PV, D_fixed, P_buy, P_sell, tasks):
    """
    PV:         array of PV generation per tick (length 60)
    D_fixed:    array of non-deferrable demand per tick (length 60)
    P_buy:      array of buy-price per tick (length 60)
    P_sell:     array of sell-price per tick (unused in this greedy)
    tasks:      list of tuples (E_i, t_start, t_end) where:
                  E_i       = energy to schedule (in same units) over the interval
                  t_start   = starting tick (inclusive)
                  t_end     = ending tick   (exclusive)

    Returns:
      D_def_profile: length-60 array, deferrable demand per tick (in original units)
      schedule:       list of per-task plans, each a list of (tick, alloc) pairs
    """
    T = 60
    # D_def stores deferrable demand in kW; multiplied by 5 yields energy in same units as E_i
    D_def = np.zeros(T)
    schedule = []

    for E_i, t_start, t_end in tasks:
        # build the candidate tick range
        time_range = list(range(t_start, t_end))
        tick_scores = []

        # score each tick: negative if PV surplus, else by buy-price
        for t in time_range:
            net_surplus = PV[t] - D_fixed[t]
            score = -net_surplus if net_surplus > 0 else P_buy[t]
            tick_scores.append((t, score))

        # sort ticks by ascending score (best slots first)
        tick_scores.sort(key=lambda x: x[1])

        remaining = E_i
        plan = []

        # allocate greedily into sorted ticks
        for t, _ in tick_scores:
            if remaining <= 0:
                break

            net_surplus = PV[t] - D_fixed[t]
            # available power in kW: 4 if surplus, else 2.5
            available_power = 4.0 if net_surplus > 0 else 2.5
            # cap so total energy in tick ≤ 20 units (over 5-minute interval)
            max_energy = max(0, 20.0 - (D_fixed[t] + D_def[t]) * 5)

            # allocate = min(available_power*5, remaining, max_energy)
            alloc = min(available_power * 5, remaining, max_energy)
            # update per-tick deferrable demand in kW
            D_def[t] += alloc / 5
            remaining -= alloc
            plan.append((t, alloc))

        schedule.append(plan)

    # return energy per tick (kW * 5) and per-task plans
    return D_def * 5, schedule
