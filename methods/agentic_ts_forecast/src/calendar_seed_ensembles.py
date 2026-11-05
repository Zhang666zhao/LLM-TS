from common import *
from blend_sources import run
names=['20261105_transformedcalft_random_mse']+[f'20261105_calendarvariant_seed{s}_mse' for s in (2022,2023,2024)]
for n in (2,3,4):run({'sources':{'run:'+k:1/n for k in names[:n]}},f'20261105_calendarseed_ensemble{n}_mse')
