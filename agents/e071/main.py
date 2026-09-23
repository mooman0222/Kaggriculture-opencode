"""Kaggriculture agent: group-manager executor + scripted market (no model)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import numpy as np


class NPPolicy:
    def __init__(self, *a, **k):
        raise RuntimeError("model required")

def encode_obs(*a, **k):
    raise RuntimeError("model required")

"""Live BC agent (numpy) for kaggriculture."""


OPS = ['PASS','NORTH','SOUTH','EAST','WEST','PICKUP','DROP','PLACE','PLANT','WATER',
       'HARVEST','FERTILIZE','DIG','BUILD_COOP','BUILD_PASTURE','FEED',
       'COLLECT_FERTILIZER','CARE']
ITEM_BY_IDX = {1:'WHEAT',2:'CARROT',3:'TOMATO',4:'STRAWBERRY',5:'MELON',6:'EGG',
               7:'MILK',8:'WOOL',9:'FERTILIZER',10:'GOOSE',11:'COW',12:'SHEEP'}
ITEM_BY_IDX_INV = {v: k for k, v in ITEM_BY_IDX.items()}
QTY_BINS = [0,1,2,3,4,5,6,7,8,9,10,12,15,20,30,50,100,1000]
QTY_REP = [0] + QTY_BINS[1:] + [1000]
LAND_PRICES = [1000, 2000, 4000]
SEED_COST = {'WHEAT':10,'CARROT':20,'TOMATO':50,'STRAWBERRY':100,'MELON':80}
ANIM_COST = {'GOOSE':300,'COW':400,'SHEEP':500}
SHED_ADJ = {(4,4),(5,4),(4,5),(5,5)}
SHED_ADJ_TILES = SHED_ADJ
# class weights used during training (weighted CE); divide out to recalibrate
MKT_W = {'bseed': [0.0049, 0.0282, 0.0548, 0.1169, 0.1473, 0.1828, 0.1634, 0.2, 0.2588, 0.31, 0.3219, 0.3265, 0.3678, 0.4145, 0.8148, 3.8218, 3.8218, 3.8218, 3.8218], 'bprod': [0.0066, 0.1271, 0.1338, 0.1473, 0.2591, 0.208, 0.3044, 0.3883, 0.3659, 0.3983, 0.3279, 0.3209, 0.2715, 0.2271, 0.5307, 0.4577, 0.5233, 7.001, 7.001], 'banim': [0.0027, 0.0355, 0.0537, 0.0983, 0.2099, 0.2634, 0.3635, 0.4809, 1.6659, 1.6659, 1.6659, 1.6659, 1.6659, 1.6659, 0.833, 1.6659, 1.6659, 1.6659, 1.6659], 'sell': [0.0128, 0.113, 0.1331, 0.1597, 0.1711, 0.2771, 0.1705, 0.3632, 0.2738, 0.3316, 0.3632, 0.2797, 0.3551, 0.3615, 0.4342, 0.7493, 0.8818, 0.3127, 8.0]}
CROPS = ['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON']
PRODUCTS = ['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER']
ANIMALS = ['GOOSE','COW','SHEEP']
CROP_FIRST_YIELD = {'WHEAT':2,'CARROT':2,'TOMATO':8,'STRAWBERRY':10,'MELON':10}
CROP_MAX_YIELD_DAY = {'WHEAT':4,'CARROT':3,'TOMATO':8,'STRAWBERRY':10,'MELON':12}
CROP_MAXY = {'WHEAT': 4, 'CARROT': 3, 'TOMATO': 4, 'STRAWBERRY': 4, 'MELON': 6}
CROP_INTERVAL = {'TOMATO': 1, 'STRAWBERRY': 2}
ANIM_MAXHELD = {'GOOSE': 4, 'COW': 6, 'SHEEP': 6}
ANIM_STRUCT = {'GOOSE':'COOP','COW':'PASTURE','SHEEP':'PASTURE'}

class Agent:
    def __init__(self, npz=None, ctx=32, temp=0.0, style=0, sample=False, seed=None, use_task=True, use_weights=True):
        self.pol = NPPolicy(npz, ctx_len=ctx) if npz else None
        self.ctx = ctx
        self.temp = temp
        self.style = style
        self.sample = sample
        self.rng = np.random.RandomState(seed)
        self.tasks = {}
        self.last_day = -1
        self.use_task = use_task
        self.last_acts = np.zeros((16, 2), dtype=np.int64)
        self.use_weights = use_weights
        self.executor_only = False
        self.hist = []
        self.last_step = -1
        self.ptask = {}

    def reset(self):
        self.hist = []
        self.hist_acts = []
        self.last_step = -1
        self.last_acts = np.zeros((16, 2), dtype=np.int64)
        self.ptask = {}

    def sticky_valid(self, tile, inv):
        """Does this tile still need a visit (mirror of on-tile actions)?"""
        if not isinstance(tile, dict):
            return False
        if tile.get('kind') == 'WEED':
            return True
        if self._harvestable(tile, getattr(self, '_day', 0)):
            return True
        if tile.get('kind') == 'PLANT' and not tile.get('watered_today'):
            return True
        if 'animal' in tile:
            if tile.get('yield_units', 0) > 0:
                return True
            if not tile.get('fed_today') and inv.get('WHEAT', 0) > 0:
                return True
            if not tile.get('cared_today'):
                return True
            if tile.get('fertilizer_available'):
                return True
        return False

    def __call__(self, obs):
        if self.pol is None:
            return self._exec_call(obs)
        step = int(obs.get('step', 0))
        if step == 0 or step <= self.last_step:
            self.reset()
        self.last_step = step
        try:
            tiles, units, items, glob = encode_obs(obs, step)
            h = self.pol.encode_step(tiles, units, items, glob)
        except Exception:
            return {'farmer': ['PASS'], 'hands': [], 'market': []}
        self.hist.append(h)
        af = np.zeros((16, 31), dtype=np.float32)
        for ui in range(16):
            af[ui, int(self.last_acts[ui, 0])] = 1.0
            af[ui, 18 + int(self.last_acts[ui, 1])] = 1.0
        self.hist_acts.append(af)
        if len(self.hist) > self.ctx:
            self.hist = self.hist[-self.ctx:]
            self.hist_acts = self.hist_acts[-self.ctx:]
        try:
            acts_hist = np.stack(self.hist_acts) if self.hist_acts else None
            out = self.pol.forward(np.stack(self.hist), style=self.style, acts=acts_hist)
        except Exception:
            return {'farmer': ['PASS'], 'hands': [], 'market': []}
        try:
            act = self.decode(out, obs)
            self._remember(act)
            return act
        except Exception as e:
            import os as _os
            if _os.environ.get('BC_DEBUG'):
                import traceback; traceback.print_exc()
            return {'farmer': ['PASS'], 'hands': [], 'market': []}

    # ---------- decoding ----------
    @staticmethod
    def _arg(x):
        return int(np.argmax(x))

    def decode(self, out, obs, use_task=None):
        if use_task is None:
            use_task = self.use_task
        self._invs = obs['private']['inventories']
        self._day = int(obs['day'])
        self._planted = set()
        self._claimed = set()
        me = int(obs.get('player', 0))
        day0 = int(obs['day'])
        if day0 != self.last_day:
            self.tasks = {}
            self.last_day = day0
        farm = obs['farms'][me]
        priv = obs['private']
        day = int(obs['day'])
        roster = [farm['farmer']] + list(farm['hands'])
        acts = []
        for i, pos in enumerate(roster[:16]):
            if use_task:
                a = self.dec_unit_task(out['dop'][i], out['dest'][i], out['darg'][i],
                                       pos, farm, priv, day)
            else:
                a = self.dec_unit(out['fop'][i], out['farg'][i], pos, farm, priv, day)
            acts.append(a)
        farmer = acts[0] if acts else ['PASS']
        hands = acts[1:]
        market = self.dec_market(out, obs, me)
        return {'farmer': farmer, 'hands': hands, 'market': market}

    @staticmethod
    def _step_toward(farm, start, goal):
        if start == goal:
            return None
        if goal[0] > start[0]:
            return 'EAST'
        if goal[0] < start[0]:
            return 'WEST'
        if goal[1] > start[1]:
            return 'SOUTH'
        if goal[1] < start[1]:
            return 'NORTH'
        return None

    def _find_tile(self, farm, pred, start=None):
        tiles = [(tx, ty) for ty in range(10) for tx in range(10) if pred(farm['tiles'][ty][tx])]
        if not tiles:
            return None
        if start is not None:
            tiles.sort(key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]))
        return tiles[0]

    def _remember(self, act):
        self.last_acts = np.zeros((16, 2), dtype=np.int64)
        roster = [act.get('farmer') or ['PASS']] + list(act.get('hands') or [])
        for i, a in enumerate(roster[:16]):
            if not a:
                continue
            op = a[0]
            self.last_acts[i, 0] = OPS.index(op) if op in OPS else 0
            arg = 0
            if len(a) > 1 and isinstance(a[1], str):
                arg = ITEM_BY_IDX_INV.get(a[1], 0)
            self.last_acts[i, 1] = arg

    def _exec_call(self, obs):
        """Pure-executor call: no model forward (for fast, model-free submission)."""
        step = int(obs.get('step', 0))
        if step == 0 or step <= self.last_step:
            self.tasks = {}
            self.last_day = -1
            self.ptask = {}
        self.last_step = step
        self._invs = obs['private']['inventories']
        self._day = int(obs['day'])
        self._planted = set()
        self._claimed = set()
        me = int(obs.get('player', 0))
        day0 = int(obs['day'])
        if day0 != self.last_day:
            self.tasks = {}
            self.last_day = day0
        farm = obs['farms'][me]
        priv = obs['private']
        roster = [farm['farmer']] + list(farm['hands'])
        acts = []
        for i, pos in enumerate(roster[:16]):
            fb = self.fallback_task(pos, farm, priv, i)
            acts.append(fb if fb is not None else ['PASS'])
        return {'farmer': acts[0] if acts else ['PASS'], 'hands': acts[1:], 'market': []}

    def fallback_task(self, pos, farm, priv, idx):
        """Farm manager: nearest unclaimed urgent task; plant to target; weed; animal logistics."""
        x, y = int(pos[0]), int(pos[1])
        inv = {}
        invs = getattr(self, '_invs', None)
        if invs is not None and idx is not None and idx < len(invs):
            inv = invs[idx]
        day = getattr(self, '_day', 0)
        tile = farm['tiles'][y][x]
        hands = 1 + len(farm['hands'])
        # --- role split: dedicated herders service animals, field hands farm ---
        roster_n = 1 + len(farm['hands'])
        n_herders = int(getattr(self, 'n_herders', 0) or 0)
        n_animals_here = sum(1 for row in farm['tiles'] for t in row if isinstance(t, dict) and 'animal' in t)
        herding_active = n_herders > 0 and n_animals_here >= 3
        is_herder = herding_active and idx is not None and idx > 0 and idx >= roster_n - n_herders
        # --- act on current tile ---
        if isinstance(tile, dict):
            if tile.get('kind') == 'WEED':
                return ['DIG']
            if self._harvestable(tile, day):
                return ['HARVEST']
            if tile.get('kind') == 'PLANT' and not tile.get('watered_today'):
                return ['WATER']
            if 'animal' in tile:
                if tile.get('yield_units', 0) > 0:
                    return ['HARVEST']
                if not tile.get('fed_today') and inv.get('WHEAT', 0) > 0:
                    return ['FEED']
                if not tile.get('cared_today'):
                    return ['CARE']
                if tile.get('fertilizer_available'):
                    return ['COLLECT_FERTILIZER']
        # --- carrying an animal ---
        carried_animal = next((a for a in ANIMALS if inv.get(a, 0) > 0), None)
        if carried_animal is not None:
            st = ANIM_STRUCT[carried_animal]
            if isinstance(tile, dict) and tile.get('kind') == st and 'animal' not in tile:
                return ['PLACE', carried_animal, 1]
            tgt = self._find_tile(farm, lambda t, st=st: isinstance(t, dict) and t.get('kind') == st and 'animal' not in t, (x, y))
            if tgt is not None:
                s = self._step_toward(farm, (x, y), tgt)
                if s:
                    return [s]
            # no empty home: designate a clustered build site (adjacent to an
            # existing structure, else near the shed) so pastures stay compact
            # and service trips stay short.
            if getattr(self, 'cluster_struct', True):
                claimed0 = getattr(self, '_claimed', None) or set()
                struct_adj = set()
                for sy in range(10):
                    for sx in range(10):
                        tt = farm['tiles'][sy][sx]
                        if isinstance(tt, dict) and tt.get('kind') in ('COOP', 'PASTURE'):
                            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                nx, ny = sx + dx, sy + dy
                                if 0 <= nx <= 9 and 0 <= ny <= 9 and farm['tiles'][ny][nx] is None:
                                    struct_adj.add((nx, ny))
                cands = [p for p in struct_adj if p not in claimed0]
                if not cands:
                    cands = [(tx, ty) for ty in range(10) for tx in range(10)
                             if farm['tiles'][ty][tx] is None and (tx, ty) not in claimed0]
                if cands:
                    ds = min(cands, key=lambda p: abs(p[0] - 4.5) + abs(p[1] - 4.5))
                    if (x, y) == ds:
                        return ['BUILD_PASTURE' if st == 'PASTURE' else 'BUILD_COOP']
                    claimed0.add(ds)
                    self._claimed = claimed0
                    s = self._step_toward(farm, (x, y), ds)
                    if s:
                        return [s]
                    return ['BUILD_PASTURE' if st == 'PASTURE' else 'BUILD_COOP'] if tile is None else None
            if tile is None:
                return ['BUILD_PASTURE' if st == 'PASTURE' else 'BUILD_COOP']
        # --- shed logistics ---
        if (x, y) in SHED_ADJ:
            invs = getattr(self, '_invs', []) or []
            # feed first: unfed animals escape in 2 days; new-animal pickup must not starve them.
            # when herders are active, only herders ferry wheat (clean role split).
            n_unfed_here = sum(1 for row in farm['tiles'] for t in row if isinstance(t, dict) and 'animal' in t and not t.get('fed_today'))
            if inv.get('WHEAT', 0) == 0 and priv['shed'].get('WHEAT', 0) > 0 and n_unfed_here > 0:
                n_wcarry = sum(iv.get('WHEAT', 0) for iv in invs)
                if n_wcarry < n_unfed_here and (is_herder or not herding_active):
                    return ['PICKUP', 'WHEAT', 2]
            n_acarry = sum(1 for iv in invs if any(iv.get(a, 0) > 0 for a in ANIMALS))
            max_acarry = getattr(self, 'max_acarry', 2)
            for a in ANIMALS:
                if priv['shed'].get(a, 0) > 0 and n_acarry < max_acarry:
                    st = ANIM_STRUCT[a]
                    if self._find_tile(farm, lambda t, st=st: isinstance(t, dict) and t.get('kind') == st and 'animal' not in t) is not None:
                        return ['PICKUP', a, 1]
                    # no empty home: pick up anyway to build one (BUILD is free)
                    if self._find_tile(farm, lambda t: t is None) is not None:
                        return ['PICKUP', a, 1]

            # NOTE: no DROP shuttling: dawn drop_inventories auto-deposits carried
            # goods to the shed. Manual DROPs only waste trips.
        # --- farm-wide tasks with claiming (roles from above) ---
        n_plants = 0
        n_thirsty = 0
        for row in farm['tiles']:
            for t in row:
                if isinstance(t, dict) and t.get('kind') == 'PLANT' and not t.get('watered_today'):
                    if not self._harvestable(t, day):
                        n_thirsty += 1
        triage = n_thirsty > int(getattr(self, 'triage_n', 999) or 999)
        n_plants = 0
        tasks = []
        for ty in range(10):
            for tx in range(10):
                t = farm['tiles'][ty][tx]
                if isinstance(t, dict):
                    if t.get('kind') == 'PLANT':
                        n_plants += 1
                        if is_herder:
                            continue
                        if self._harvestable(t, day):
                            tasks.append(((tx, ty), 6, 'H'))
                        elif not t.get('watered_today'):
                            tasks.append(((tx, ty), int(getattr(self, 'prio_water', 5)), 'W'))
                    elif 'animal' in t:
                        if herding_active and not is_herder:
                            continue
                        if t.get('yield_units', 0) > 0:
                            tasks.append(((tx, ty), 6, 'H'))
                        elif not t.get('fed_today') and inv.get('WHEAT', 0) > 0:
                            tasks.append(((tx, ty), 7, 'F'))
                        elif not t.get('cared_today') and not triage:
                            tasks.append(((tx, ty), 4, 'R'))
                        elif t.get('fertilizer_available') and not triage:
                            tasks.append(((tx, ty), 5, 'C'))
                    elif t.get('kind') == 'WEED':
                        if is_herder:
                            continue
                        tasks.append(((tx, ty), 3, 'D'))
        # planting targets
        cap = max(6, int(getattr(self, 'cap_mult', 2) * hands) + 2)
        cap = min(cap, getattr(self, 'cap_max', 99))
        free_tiles = [(tx, ty) for ty in range(10) for tx in range(10) if farm['tiles'][ty][tx] is None]
        seeds = priv['seeds']
        want = []
        if n_plants < cap and not is_herder:
            order = getattr(self, 'crop_order', ('MELON', 'STRAWBERRY', 'WHEAT', 'CARROT'))
            if day >= getattr(self, 'late_day', 99) and hasattr(self, 'late_order'):
                order = self.late_order
            for c in order:
                if seeds.get(c, 0) > 0:
                    want.append(c)
        claimed = getattr(self, '_claimed', None)
        if claimed is None:
            claimed = set()
            self._claimed = claimed
        ptask = getattr(self, 'ptask', None)
        if ptask is None:
            ptask = {}
            self.ptask = ptask
        # sticky target: keep walking to last turn's goal if it still needs us
        # (kills ping-pong oscillation between equally-good targets)
        if getattr(self, 'sticky', True) and idx in ptask:
            stx, sty = ptask[idx]
            in_board = 0 <= stx <= 9 and 0 <= sty <= 9
            if in_board and (stx, sty) != (x, y) and (stx, sty) not in claimed:
                if self.sticky_valid(farm['tiles'][sty][stx], inv):
                    claimed.add((stx, sty))
                    s = self._step_toward(farm, (x, y), (stx, sty))
                    if s:
                        return [s]
                    ptask.pop(idx, None)
                else:
                    ptask.pop(idx, None)
            else:
                ptask.pop(idx, None)
        if tasks:
            tasks.sort(key=lambda z: (-z[1], abs(z[0][0] - x) + abs(z[0][1] - y)))
            for tgt, prio, kind in tasks:
                if tgt == (x, y):
                    break
                if tgt in claimed:
                    continue
                claimed.add(tgt)
                ptask[idx] = tgt
                s = self._step_toward(farm, (x, y), tgt)
                if s:
                    return [s]
                break
        if want and free_tiles:
            if tile is None:
                c = want[0]
                planted = getattr(self, '_planted', None)
                if planted is None:
                    planted = set()
                    self._planted = planted
                if c not in planted:
                    planted.add(c)
                    return ['PLANT', c]
            free_unclaimed = [p for p in free_tiles if p not in claimed]
            if free_unclaimed:
                if getattr(self, 'compact_plant', False):
                    tgt = min(free_unclaimed, key=lambda p: abs(p[0] - 4.5) + abs(p[1] - 4.5))
                else:
                    tgt = min(free_unclaimed, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
                claimed.add(tgt)
                s = self._step_toward(farm, (x, y), tgt)
                if s:
                    return [s]
        # --- idle: walk home to the shed area to stay compact ---
        if (x, y) not in SHED_ADJ:
            tgt = min(SHED_ADJ, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
            s = self._step_toward(farm, (x, y), tgt)
            if s:
                return [s]
        return None

    def _prob(self, logits, w=None):
        z = np.asarray(logits, dtype=np.float64)
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        if w is not None:
            p = p / np.asarray(w, dtype=np.float64)
            p = p / p.sum()
        return p

    # ---- task-based unit decoding with persistent tasks ----
    @staticmethod
    def _harvestable(t, day):
        # FULL-YIELD harvesting: yield grows by watering during the window
        # (non-ongoing) or by daily ticks (ongoing/animals). Grabbing at first
        # maturity yields 1 unit; waiting yields up to 4-6x. Only salvage when
        # decaying/dying or at game end.
        if not isinstance(t, dict):
            return False
        if t.get('yield_units', 0) <= 0:
            return False
        if t.get('kind') == 'PLANT':
            crop = t.get('crop')
            age = day - t.get('planted_day', day)
            if age < CROP_FIRST_YIELD.get(crop, 99):
                return False
            if day >= 27:
                return True  # endgame: take everything
            if crop in CROP_INTERVAL:
                maxy = CROP_MAXY.get(crop, 4)
                if t.get('yield_units', 0) >= maxy:
                    return True
                since = age - CROP_FIRST_YIELD.get(crop, 99)
                count = since // CROP_INTERVAL[crop] + 1
                return count >= 4  # last cycle: salvage before death
            maxy = CROP_MAXY.get(crop, 4)
            if t.get('yield_units', 0) >= maxy:
                return True
            return age > CROP_MAX_YIELD_DAY.get(crop, 99)  # decaying: salvage
        if 'animal' in t:
            if day >= 28:
                return True
            return t.get('yield_units', 0) >= ANIM_MAXHELD.get(t.get('animal'), 6)
        return True

    def feasible(self, op, tile, arg, farm, priv):
        if op == 'PICKUP':
            return tile in SHED_ADJ_TILES and priv['shed'].get(arg, 0) > 0
        if op == 'DROP':
            return tile in SHED_ADJ_TILES
        if op == 'PLACE':
            if arg in ANIMALS:
                return 0 <= tile[0] <= 9 and 0 <= tile[1] <= 9 and \
                    isinstance(farm['tiles'][tile[1]][tile[0]], dict) and \
                    farm['tiles'][tile[1]][tile[0]].get('kind') == ANIM_STRUCT[arg] and \
                    'animal' not in farm['tiles'][tile[1]][tile[0]]
            return tile in SHED_ADJ_TILES
        if not (0 <= tile[0] <= 9 and 0 <= tile[1] <= 9):
            return False
        t = farm['tiles'][tile[1]][tile[0]]
        if op == 'PLANT':
            return t is None and priv['seeds'].get(arg, 0) > 0
        if op == 'WATER':
            return isinstance(t, dict) and t.get('kind') == 'PLANT' and not t.get('watered_today')
        if op == 'HARVEST':
            return self._harvestable(t, getattr(self, '_day', 0))
        if op == 'FERTILIZE':
            return isinstance(t, dict) and t.get('kind') == 'PLANT'
        if op in ('BUILD_COOP', 'BUILD_PASTURE'):
            return t is None
        if op == 'FEED':
            return isinstance(t, dict) and 'animal' in t and not t.get('fed_today')
        if op == 'COLLECT_FERTILIZER':
            return isinstance(t, dict) and t.get('fertilizer_available')
        if op == 'CARE':
            return isinstance(t, dict) and 'animal' in t and not t.get('cared_today')
        if op == 'DIG':
            return t == 'WEED' or (isinstance(t, dict) and t.get('kind') == 'PLANT')
        return False

    def do_here(self, op, arg, pos, farm, priv):
        x, y = int(pos[0]), int(pos[1])
        tile = farm['tiles'][y][x]
        if not self.feasible(op, (x, y), arg, farm, priv):
            return None
        if op == 'PICKUP':
            return ['PICKUP', arg, min(priv['shed'].get(arg, 0), 1 if arg in ANIMALS else 3)]
        if op == 'DROP':
            return ['DROP']
        if op == 'PLACE':
            return ['PLACE', arg, 1]
        if op == 'PLANT':
            return ['PLANT', arg]
        if op == 'DIG':
            return ['DIG']
        return [op]

    def cont_task(self, tk, pos, farm, priv):
        op, (tx, ty), arg = tk['op'], tk['dest'], tk['arg']
        x, y = int(pos[0]), int(pos[1])
        if (x, y) == (tx, ty):
            tk['done'] = True
            return self.do_here(op, arg, pos, farm, priv)
        if not self.feasible(op, (tx, ty), arg, farm, priv):
            return None
        step = self._step_toward(farm, (x, y), (tx, ty))
        return [step] if step else None

    def dec_unit_task(self, dop_logits, dest_logits, darg_logits, pos, farm, priv, day, idx=None):
        if self.executor_only:
            fb = self.fallback_task(pos, farm, priv, idx)
            return fb if fb is not None else ['PASS']
        if idx is not None:
            tk = self.tasks.get(idx)
            if tk is not None:
                act = self.cont_task(tk, pos, farm, priv)
                if act is not None:
                    tk['age'] += 1
                    if tk.get('done') or tk['age'] > 40:
                        self.tasks.pop(idx, None)
                    return act
                self.tasks.pop(idx, None)
        p_dop = self._prob(dop_logits)
        dt = np.argsort(-p_dop)[:8]
        et = np.argsort(-dest_logits)[:8]
        at = np.argsort(-darg_logits)[:8]
        cand = []
        scores = []
        for d in dt:
            d = int(d)
            if d == 0:
                cand.append(('PASS', None, None))
                scores.append(float(np.log(p_dop[0] + 1e-9)))
                continue
            op = OPS[d - 1]
            for e in et:
                e = int(e)
                if e == 0:
                    continue
                ev = e - 1
                tile = (ev % 10, ev // 10)
                for a in at:
                    a = int(a)
                    arg = None
                    if op in ('PICKUP', 'PLACE', 'PLANT'):
                        arg = ITEM_BY_IDX.get(a)
                        if arg is None:
                            continue
                    if not self.feasible(op, tile, arg, farm, priv):
                        continue
                    cand.append((op, tile, arg))
                    scores.append(float(np.log(p_dop[d] + 1e-9)) + float(dest_logits[e]) + float(darg_logits[a]))
        if not cand:
            return ['PASS']
        if not self.sample:
            k = int(np.argmax(scores))
        else:
            s = np.asarray(scores, dtype=np.float64)
            s = s - s.max()
            prob = np.exp(s)
            prob = prob / prob.sum()
            k = int(self.rng.choice(len(cand), p=prob))
        op, tile, arg = cand[k]
        if op == 'PASS':
            fb = self.fallback_task(pos, farm, priv, idx)
            if fb is not None:
                return fb
            return ['PASS']
        if idx is not None:
            self.tasks[idx] = {'op': op, 'dest': tile, 'arg': arg, 'age': 0}
        act = self.do_here(op, arg, pos, farm, priv) if tile == (int(pos[0]), int(pos[1])) else None
        if act is not None:
            if idx is not None:
                self.tasks.pop(idx, None)
            return act
        step = self._step_toward(farm, (int(pos[0]), int(pos[1])), tile)
        if step is None:
            if idx is not None:
                self.tasks.pop(idx, None)
            return ['PASS']
        return [step]

    def dec_unit(self, fop, farg, pos, farm, priv, day):
        cls = self._arg(fop)
        if cls == 0:
            return ['PASS']
        op = OPS[cls - 1]
        x, y = int(pos[0]), int(pos[1])
        tile = farm['tiles'][y][x]
        shed_adj = (x, y) in SHED_ADJ
        shed = priv['shed']
        inv = None
        arg = None
        if cls - 1 in (5, 7, 8):
            ai = self._arg(farg)
            arg = ITEM_BY_IDX.get(ai)
            if arg is None:
                return ['PASS']
        if op == 'PICKUP':
            if not shed_adj or shed.get(arg, 0) <= 0:
                return ['PASS']
            n = min(shed.get(arg, 0), 1 if arg in ANIMALS else 3)
            return ['PICKUP', arg, n]
        if op == 'PLACE':
            if arg in ANIMALS:
                if isinstance(tile, dict) and tile.get('kind') == ANIM_STRUCT[arg] and 'animal' not in tile:
                    return ['PLACE', arg, 1]
                return ['PASS']
            if shed_adj:
                return ['PLACE', arg, 1]
            return ['PASS']
        if op == 'PLANT':
            if tile is None and priv['seeds'].get(arg, 0) > 0:
                return ['PLANT', arg]
            return ['PASS']
        if op == 'WATER':
            if isinstance(tile, dict) and tile.get('kind') == 'PLANT' and not tile.get('watered_today'):
                return ['WATER']
            return ['PASS']
        if op == 'HARVEST':
            if isinstance(tile, dict) and tile.get('yield_units', 0) > 0:
                return ['HARVEST']
            return ['PASS']
        if op == 'FERTILIZE':
            if isinstance(tile, dict) and tile.get('kind') == 'PLANT':
                return ['FERTILIZE']
            return ['PASS']
        if op in ('BUILD_COOP', 'BUILD_PASTURE'):
            if tile is None and float(farm['money']) >= 300:
                return [op]
            return ['PASS']
        if op == 'FEED':
            if isinstance(tile, dict) and 'animal' in tile and not tile.get('fed_today'):
                return ['FEED']
            return ['PASS']
        if op == 'COLLECT_FERTILIZER':
            if isinstance(tile, dict) and tile.get('fertilizer_available'):
                return ['COLLECT_FERTILIZER']
            return ['PASS']
        if op == 'CARE':
            if isinstance(tile, dict) and 'animal' in tile and not tile.get('cared_today'):
                return ['CARE']
            return ['PASS']
        if op == 'DIG':
            if tile == 'WEED' or (isinstance(tile, dict) and tile.get('kind') == 'PLANT'):
                return ['DIG']
            return ['PASS']
        if op == 'DROP':
            if shed_adj:
                return ['DROP']
            return ['PASS']
        if op in ('NORTH', 'SOUTH', 'EAST', 'WEST'):
            return [op]
        return ['PASS']

    def cal(self, logits, w=None):
        # training used weighted CE with weights w; p_model ~ w * P_data -> divide by w
        if not self.use_weights:
            w = None
        p = self._prob(logits, w)
        if not self.sample:
            return int(np.argmax(p))
        return int(self.rng.choice(len(p), p=p))

    def dec_market(self, out, obs, me):
        farm = obs['farms'][me]
        shed = obs['private']['shed']
        money = float(farm['money'])
        cand = []
        n_hire = self._arg(out['hire'])
        nq = len(farm['unlocked_quadrants'])
        if n_hire > 0:
            cand.append((0, [['HIRE']] * min(n_hire, 6)))
        if self._arg(out['land']) == 1 and nq < 4 and money >= LAND_PRICES[nq - 1]:
            cand.append((1, [['BUY_LAND']]))
        for i, c in enumerate(CROPS):
            q = QTY_REP[self.cal(out['bseed'][i], MKT_W['bseed'])]
            if q > 0 and money >= SEED_COST[c]:
                cand.append((3, [['BUY_SEED', c, min(q, int(money // SEED_COST[c]))]]))
        for i, a in enumerate(ANIMALS):
            q = QTY_REP[self.cal(out['banim'][i], MKT_W['banim'])]
            if q > 0 and money >= ANIM_COST[a]:
                cand.append((2, [['BUY_ANIMAL', a, min(q, int(money // ANIM_COST[a]))]]))
        for i, p in enumerate(PRODUCTS):
            q = QTY_REP[self.cal(out['bprod'][i], MKT_W['bprod'])]
            if q > 0 and money >= 20:
                cand.append((5, [['BUY_PRODUCT', p, min(q, max(1, int(money // 25)))]]))
        for i, p in enumerate(PRODUCTS):
            q = QTY_REP[self.cal(out['sell'][i], MKT_W['sell'])]
            have = shed.get(p, 0)
            if q > 0 and have > 0:
                cand.append((1, [['SELL', p, min(q, have)]]))
        cand.sort(key=lambda t: t[0])
        orders = []
        for _, lst in cand:
            for o in lst:
                if len(orders) >= 10:
                    break
                orders.append(o)
            if len(orders) >= 10:
                break
        return orders

"""Scripted market + executor v2 evaluation."""
sys.path.insert(0, '/home/mooman_che/Kaggriculture-opencode/third_party/kaggriculture-cppsim')
sys.path.insert(0, '/tmp/t10')

BASE = {'WHEAT':25,'CARROT':35,'TOMATO':60,'STRAWBERRY':120,'MELON':250,'EGG':50,'MILK':160,'WOOL':200,'FERTILIZER':100}
SEEDC = {'WHEAT':10,'CARROT':20,'TOMATO':50,'STRAWBERRY':100,'MELON':80}
ANIMC = {'GOOSE':300,'COW':400,'SHEEP':500}
FIB = [1,1,2,3,5,8,13,21,34,55]

def market(o):
    me = int(o.get('player', 0))
    f = o['farms'][me]; priv = o['private']; money = float(f['money'])
    day = int(o['day'])
    animals = sum(1 for row in f['tiles'] for t in row if isinstance(t, dict) and 'animal' in t)
    orders = []
    # sell FIRST (income engine; never starve these)
    prices = o['market']['prices']
    sell_px = getattr(market, 'sell_px', 0.95)
    dump_n = getattr(market, 'dump_n', 12)
    for p, have in list(priv['shed'].items()):
        if p in BASE and have > 0:
            qty = have - (2 * animals if p == 'WHEAT' and animals > 0 else 0)
            if qty <= 0:
                continue
            if day >= 28 or prices[p] >= sell_px * BASE[p] or have >= dump_n:
                orders.append(['SELL', p, qty])
    # feed wheat: deficit-based, keep 2 days reserve; never auto-sell the reserve
    if animals > 0 and money > 100:
        reserve = 2 * animals
        q = reserve - priv['shed'].get('WHEAT', 0)
        if q > 0:
            orders.append(['BUY_PRODUCT', 'WHEAT', q])
    # hire (leave >=2 slots for seeds/land/animals below)
    n = min(getattr(market, 'max_hire', 8), 12 - f['hires_today'], max(0, 8 - len(orders)))
    m = money
    for i in range(max(0, n)):
        c = FIB[min(f['hires_today'] + i, 9)]
        if m < c or c > 0.2 * money:
            break
        m -= c
        orders.append(['HIRE'])
    # seeds to target
    seeds = priv['seeds']
    plants = sum(1 for row in f['tiles'] for t in row if isinstance(t, dict) and t.get('kind') == 'PLANT')
    hands = 1 + len(f['hands'])
    cap = max(6, 2 + 2 * hands)
    seed_targets = getattr(market, 'seed_targets', (('MELON', 6), ('STRAWBERRY', 4), ('WHEAT', 4)))
    if day >= getattr(market, 'late_day', 99) and hasattr(market, 'seed_targets_late'):
        seed_targets = market.seed_targets_late
    if plants < cap and money > 400:
        for c, target in seed_targets:
            q = target - seeds.get(c, 0)
            if q > 0 and money > q * SEEDC[c] + 200:
                orders.append(['BUY_SEED', c, q])
    # land
    nq = len(f['unlocked_quadrants'])
    LAND = [1000, 2000, 4000]
    if nq < getattr(market, 'max_quad', 4) and money > LAND[nq - 1] * getattr(market, 'land_mult', 2):
        orders.append(['BUY_LAND'])
    # animals: configurable targets {type: (max_count, day_from, day_to, money_min)}
    animals = sum(1 for row in f['tiles'] for t in row if isinstance(t, dict) and 'animal' in t)
    shed_animals = sum(priv['shed'].get(a, 0) for a in ANIMC)
    aplan = getattr(market, 'animal_plan', None)
    if aplan is None:
        if money > 4000 and animals + shed_animals < 2:
            orders.append(['BUY_ANIMAL', 'COW', 1])
            orders.append(['BUY_ANIMAL', 'SHEEP', 1])
    else:
        min_feed = getattr(market, 'animal_min_feed', 0)
        unfed_now = sum(1 for row in f['tiles'] for t in row if isinstance(t, dict) and 'animal' in t and not t.get('fed_today'))
        if unfed_now == 0:
            for typ, mx, d0, d1, m0 in aplan:
                if d0 <= day <= d1 and money > m0 and priv['shed'].get('WHEAT', 0) >= min_feed:
                    have = (sum(1 for row in f['tiles'] for t in row if isinstance(t, dict) and t.get('animal') == typ)
                            + priv['shed'].get(typ, 0))
                    if have < mx:
                        orders.append(['BUY_ANIMAL', typ, 1])
    return orders[:10]


_AGENT = None

def _make():
    ag = Agent(None, ctx=32, style=1, sample=False, use_task=True, use_weights=False)
    ag.executor_only = True
    ag.cap_mult = 3
    ag.max_acarry = 6
    ag.crop_order = ("WHEAT", "MELON", "STRAWBERRY", "CARROT")
    ag.late_day = 16
    ag.late_order = ("WHEAT", "CARROT")
    market.max_hire = 8
    market.max_quad = 2
    market.late_day = 16
    market.seed_targets_late = (("WHEAT", 8),)
    market.animal_plan = [("SHEEP", 10, 4, 29, 1000), ("COW", 7, 6, 29, 1000)]
    market.animal_min_feed = 0
    return ag

def agent(observation, configuration=None):
    global _AGENT
    if _AGENT is None:
        _AGENT = _make()
    a = _AGENT(observation)
    try:
        a["market"] = market(observation)
    except Exception:
        a["market"] = []
    return a

