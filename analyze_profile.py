import pstats

p = pstats.Stats('simulate_yuda.prof')
p.sort_stats('cumtime').print_stats(30)
