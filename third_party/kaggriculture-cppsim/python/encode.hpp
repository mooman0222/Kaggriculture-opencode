// Bit-exact C++ port of rl/features.encode (+ rl/sp/legal_all.legal_all): observation tensors for the RL policy,
// built straight from the engine state (no Python dict). Layouts must stay in lockstep with rl/features.py.
#pragma once
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include "../sim/sim.hpp"

namespace enc {
using namespace kag;
constexpr int RL_MAX_UNITS = 16, TILE_F = 18, UNIT_F = 18, ITEM_F = 9, GLOB_F = 40, N_OPS = 44;
constexpr int BASE_PRICE[N_PRODUCTS] = {25, 35, 60, 120, 250, 50, 160, 200, 100};
constexpr int FIRST_YIELD[N_CROPS] = {2, 2, 8, 10, 10};
// features.SHOPS order: BAKERY, PIZZA_SHOP, BRUNCH_SPOT, YARN_STORE, ICE_CREAM_SHOP, PET_CAFE, SMOOTHIE_SHOP, FARMERS_MARKET
constexpr int SHOP_SLOT[N_SHOPS] = {0, 2, 7, 4, 5, 1, 6, 3};  // indexed by ShopId (sorted order)
constexpr int LAND_PRICES[3] = {1000, 2000, 4000};
// features.OPS indices
enum : int { O_PASS = 0, O_PLANT0 = 5, O_WATER = 10, O_HARVEST = 11, O_FERTILIZE = 12, O_DIG = 13, O_BUILD_COOP = 14, O_BUILD_PASTURE = 15,
             O_PICKUP0 = 16, O_DROP = 28, O_PLACE0 = 29, O_FEED = 41, O_CARE = 42, O_COLLECT = 43 };
inline bool is_shed(int x, int y) { return (x == 4 || x == 5) && (y == 4 || y == 5); }
inline int animal_product(int what) { return what == GOOSE ? EGG : (what == COW ? MILK : WOOL); }

struct Out { int16_t* tiles; int16_t* units; float* items; float* glob; bool* legal; int16_t* pos; float* money; };

inline void tile_row(const Tile& t, int x, int y, int day, int step, int opp, int units_here, int16_t* f) {
    std::memset(f, 0, TILE_F * sizeof(int16_t)); f[16] = x; f[17] = y; f[15] = opp; f[14] = units_here;
    switch (t.kind) {
    case T_LOCKED: f[0] = 0; return;
    case T_EMPTY: f[0] = 1; return;
    case T_WEED: f[0] = 2; return;
    case T_COOP: case T_PASTURE:
        if (t.has_animal) {
            f[0] = 6; f[2] = t.what - GOOSE + 1; f[3] = std::min(30, std::max(0, day - (int)t.planted_day)); f[4] = std::min(6, (int)t.yield_units);
            f[8] = t.fed_today; f[9] = t.cared_today; f[10] = std::min(3, (int)t.consecutive_dry); f[11] = t.fertilizer_available; f[12] = std::min(3, (int)t.pending_care_bonus);
        } else f[0] = (t.kind == T_COOP) ? 4 : 5;
        return;
    case T_PLANT: {
        f[0] = 3; f[1] = t.what + 1; f[3] = std::min(30, std::max(0, day - (int)t.planted_day)); f[4] = std::min(6, (int)t.yield_units);
        f[5] = t.watered_today; f[6] = std::min(3, (int)t.consecutive_dry); f[7] = std::max(0, std::min(3, (int)t.fertilized_until_day - day + 1));
        int mls = t.max_lifespan_step; int q = (mls - step) / 24; if ((mls - step) < 0 && (mls - step) % 24 != 0) q -= 1;  // Python floor division
        f[13] = mls < 0 ? 8 : std::max(0, std::min(8, q)); return; }
    }
}

inline void encode(const Sim& sim, int seat, Out& o) {
    const State& st = sim.st; const Farm& own = st.farms[seat]; const Farm& opp = st.farms[1 - seat];
    int step = st.step, day = step / 24, hour = step % 24; int n = std::min(own.n_units, RL_MAX_UNITS);
    int here[BOARD][BOARD]; std::memset(here, 0, sizeof(here));
    for (int i = 0; i < n; ++i) here[own.pos_y[i]][own.pos_x[i]]++;
    for (int k = 0; k < 2; ++k) { const Farm& f = k == 0 ? own : opp;
        for (int y = 0; y < BOARD; ++y) for (int x = 0; x < BOARD; ++x) tile_row(f.tiles[y][x], x, y, day, step, k, k == 0 ? here[y][x] : 0, o.tiles + ((k * 10 + y) * 10 + x) * TILE_F); }
    std::memset(o.units, 0, RL_MAX_UNITS * UNIT_F * sizeof(int16_t));
    for (int i = 0; i < RL_MAX_UNITS; ++i) o.pos[i] = -1;
    for (int i = 0; i < n; ++i) { int16_t* u = o.units + i * UNIT_F; int x = own.pos_x[i], y = own.pos_y[i];
        u[0] = 1; u[1] = i == 0; u[2] = x; u[3] = y; for (int j = 0; j < N_ITEMS; ++j) u[4 + j] = std::min<int>(60, own.inv[i][j]); u[16] = is_shed(x, y); u[17] = i; o.pos[i] = y * 10 + x; }
    int carried[N_ITEMS] = {0}; for (int i = 0; i < own.n_units; ++i) for (int j = 0; j < N_ITEMS; ++j) carried[j] += own.inv[i][j];
    int prod_own[N_PRODUCTS] = {0}, prod_opp[N_PRODUCTS] = {0};
    for (int k = 0; k < 2; ++k) { const Farm& f = k == 0 ? own : opp; int* pr = k == 0 ? prod_own : prod_opp;
        for (int y = 0; y < BOARD; ++y) for (int x = 0; x < BOARD; ++x) { const Tile& t = f.tiles[y][x];
            if (t.kind == T_PLANT) pr[t.what]++; else if ((t.kind == T_COOP || t.kind == T_PASTURE) && t.has_animal) pr[animal_product(t.what)]++; } }
    int shopscore[N_PRODUCTS] = {0};
    for (int s = 0; s < st.n_shops; ++s) { uint16_t m = SHOP_MASK[st.shops[s]]; int w = __builtin_popcount(m) == 1 ? 2 : 1; for (int p = 0; p < N_PRODUCTS; ++p) if (m & (1u << p)) shopscore[p] += w; }
    for (int j = 0; j < N_PRODUCTS; ++j) { float* I = o.items + j * ITEM_F;
        I[0] = (float)((double)st.market.prices[j] / BASE_PRICE[j]); I[1] = (float)((st.market.inventory[j] - 10000) / 100.0); I[2] = (float)(std::min<int>(200, own.shed[j]) / 50.0); I[3] = (float)(std::min(200, carried[j]) / 50.0);
        I[4] = (float)(shopscore[j] / 4.0); I[5] = (float)(prod_own[j] / 20.0); I[6] = (float)(prod_opp[j] / 20.0); I[7] = (float)(BASE_PRICE[j] / 250.0); I[8] = (float)(j / 8.0); }
    float* G = o.glob; std::memset(G, 0, GLOB_F * sizeof(float));
    G[0] = (float)(day / 29.0); G[1] = (float)(hour / 23.0); G[2] = (float)std::sin(2 * M_PI * hour / 24.0); G[3] = (float)std::cos(2 * M_PI * hour / 24.0); G[4] = (float)(step / 719.0); G[5] = day >= 29 ? 1.f : 0.f;
    G[6] = (float)(std::log1p(std::max(0.0, own.money)) / 12.0); G[7] = (float)(std::log1p(std::max(0.0, opp.money)) / 12.0); G[8] = (float)(own.n_quadrants / 4.0); G[9] = (float)(opp.n_quadrants / 4.0);
    G[10] = (float)((own.n_units - 1) / 12.0); G[11] = (float)((opp.n_units - 1) / 12.0); G[12] = (float)(own.hires_today / 12.0);
    { double acc[8] = {0}; for (int s = 0; s < st.n_shops; ++s) acc[SHOP_SLOT[st.shops[s]]] += 1 / 3.0; for (int s = 0; s < 8; ++s) G[13 + s] = (float)acc[s]; }
    G[21] = (float)(st.n_shops / 8.0);
    for (int c = 0; c < N_CROPS; ++c) G[22 + c] = (float)(std::min<int>(60, own.seeds[c]) / 30.0);
    for (int a = 0; a < N_ANIMALS; ++a) G[27 + a] = (float)(std::min<int>(6, own.shed[GOOSE + a]) / 3.0);
    G[30] = (own.n_quadrants >= 1 && own.n_quadrants <= 3) ? (float)(LAND_PRICES[own.n_quadrants - 1] / 4000.0) : 0.f;
    { int s = 0; for (int p = 0; p < N_PRODUCTS; ++p) s += std::max<int>(0, own.shed[p]); G[31] = (float)(s / 100.0); }
    G[32] = (float)seat;
    o.money[0] = (float)own.money; o.money[1] = (float)opp.money;
    // legality [16][100][44]
    std::memset(o.legal, 0, RL_MAX_UNITS * 100 * N_OPS);
    for (int i = 0; i < RL_MAX_UNITS; ++i) for (int t = 0; t < 100; ++t) o.legal[(i * 100 + t) * N_OPS + O_PASS] = true;
    for (int i = 0; i < n; ++i) { const int16_t* inv = own.inv[i]; bool any = false; for (int j = 0; j < N_ITEMS; ++j) any |= inv[j] > 0;
        for (int y = 0; y < BOARD; ++y) for (int x = 0; x < BOARD; ++x) { bool* m = o.legal + (i * 100 + y * 10 + x) * N_OPS; const Tile& t = own.tiles[y][x];
            if (is_shed(x, y)) { for (int j = 0; j < N_ITEMS; ++j) if (own.shed[j] > 0) m[O_PICKUP0 + j] = true; if (any) m[O_DROP] = true; for (int j = 0; j < N_PRODUCTS; ++j) if (inv[j] > 0) m[O_PLACE0 + j] = true; }
            switch (t.kind) {
            case T_EMPTY: for (int c = 0; c < N_CROPS; ++c) if (own.seeds[c] > 0) m[O_PLANT0 + c] = true; m[O_BUILD_COOP] = m[O_BUILD_PASTURE] = m[O_DIG] = true; break;
            case T_WEED: m[O_DIG] = true; break;
            case T_PLANT: if (!t.watered_today) m[O_WATER] = true; m[O_DIG] = true;
                if (t.yield_units > 0 && day - t.planted_day >= FIRST_YIELD[t.what]) m[O_HARVEST] = true; if (inv[FERTILIZER] > 0) m[O_FERTILIZE] = true; break;
            case T_COOP: case T_PASTURE:
                if (t.has_animal) { if (inv[WHEAT] > 0 && !t.fed_today) m[O_FEED] = true; if (!t.cared_today) m[O_CARE] = true; if (t.yield_units > 0) m[O_HARVEST] = true; if (t.fertilizer_available) m[O_COLLECT] = true; }
                else { m[O_DIG] = true; for (int a = 0; a < N_ANIMALS; ++a) { bool coop = (GOOSE + a) == GOOSE; if (((t.kind == T_COOP) == coop) && inv[GOOSE + a] > 0) m[O_PLACE0 + GOOSE + a] = true; } }
                break;
            default: break; } } }
}
}  // namespace enc
