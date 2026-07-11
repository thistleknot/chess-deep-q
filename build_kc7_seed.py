"""Merge 7 seed builder (spec/knightcap-full.spec.md items 4+7).

Builds models/qlearn_kc7{,_best}.pt: an 809-dim linear ValueNet whose
- [:769] = the m4 champion's PST weights RESCALED so one pawn up evaluates to tanh ~= 0.25
  (donor local.h EVAL_SCALE calibration; scale = atanh(0.25)/current pawn worth), and whose
- [769:809] = the 40 donor features initialized from KnightCap's TRAINED coefficients
  (large_coeffs.h, unit pawn=10000), converted per-feature: w = pawns_per_feature_unit *
  atanh(0.25), White block +, Black block -. Slope-derived for mobility vectors; magnitudes
  are order-of-magnitude informed inits, not truth — training refines them.

Preconditions: models/qlearn_m4_best.pt exists (linear, 769). Failure mode: asserts on shape.
"""
import torch

P = 10000.0                              # donor coefficient unit: 1 pawn
# Calibration note (spec item 4, DROPPED like item 8): rescaling the champion so 1 pawn ->
# tanh 0.25 amplifies per-square PST noise x18 and inverts material responses (measured).
# Instead donor features are expressed in the CHAMPION'S own pawn unit (its mean learned pawn
# worth), preserving the donor's feature-to-material RATIOS without touching the PST.
A = None                                 # set at runtime to the champion pawn worth


def pawns(coeff):
    return coeff / P


def donor_weights(A):
    """Per-feature init in tanh units PER UNIT OF OUR NORMALIZED FEATURE (White sign; Black
    = -w), A = champion pawn worth. Normalizations: mobility /14, hung count /5, hung value
    /9, king ring /8, pawns /8, rooks /2, indicators 1. Mobility slopes = (v[9]-v[0])/9."""
    return {
        "bishop_pair": pawns(2018) * A,
        "mob": [0.0,                                                  # N: no donor mobility
                pawns((1128 + 1596) / 9) * 14 * A,                    # B
                pawns((801 + 1304) / 9) * 14 * A,                     # R
                pawns((524 + 1200) / 9) * 14 * A],                    # Q
        "smob": [pawns((0 + 777) / 9) * 14 * A,
                 pawns((-5 + 717) / 9) * 14 * A,
                 pawns((22 + 800) / 9) * 14 * A,
                 pawns((12 + 3005) / 9) * 14 * A],
        "hung_cnt": -pawns(2356) * 5 * A,                             # THREAT
        "hung_val": -4.5 * A,                                         # unit = 9 pawns hung; IHUNG
        "king_ring": -pawns(2999) * 2.0 * A,                          # penalty ~ half its value
        "castle": pawns(3056) * A,
        "doubled": -pawns(1411) * 8 * A,
        "isolated": -pawns(700) * 8 * A,
        "passed": pawns(3000) * 8 * A,
        "rook_open": pawns(1000) * 2 * A,
        "rook_half": pawns(300) * 2 * A,
        "connected": pawns(297) * A,
    }


def main():
    ck = torch.load("models/qlearn_m4_best.pt", map_location="cpu")
    w769 = ck["state_dict"]["head.weight"].reshape(-1)
    assert w769.numel() == 769, "champion must be the 769 linear net"
    pawn_worth = float((w769[0:64].mean() - w769[384:448].mean()))    # white minus black pawn planes
    assert pawn_worth > 1e-6, f"champion pawn worth degenerate: {pawn_worth}"
    W = donor_weights(pawn_worth)                                     # donor ratios, champion units
    w = torch.zeros(1, 809)
    w[0, :769] = w769                                                 # PST untouched (see note)
    def put(idx_w, idx_b, val):
        w[0, idx_w], w[0, idx_b] = val, -val
    put(769, 770, W["bishop_pair"])
    for ti in range(4):
        put(771 + ti, 775 + ti, W["mob"][ti])
        put(779 + ti, 783 + ti, W["smob"][ti])
    put(787, 788, W["hung_cnt"])
    put(789, 790, W["hung_val"])
    put(791, 792, W["king_ring"])
    put(793, 795, W["castle"])                                        # WK/BK
    put(794, 796, W["castle"])                                        # WQ/BQ
    put(797, 798, W["doubled"])
    put(799, 800, W["isolated"])
    put(801, 802, W["passed"])
    put(803, 804, W["rook_open"])
    put(805, 806, W["rook_half"])
    put(807, 808, W["connected"])
    sd = {"head.weight": w, "head.bias": ck["state_dict"]["head.bias"].clone()}
    out = {"state_dict": sd, "arch": "linear", "enc": "kc",
           "cum_games": int(ck.get("cum_games", 4200))}
    for p in ("models/qlearn_kc7.pt", "models/qlearn_kc7_best.pt"):
        torch.save(out, p)
    print(f"kc7 seed: pawn_worth {pawn_worth:.4f} (unit for donor ratios); donor features set; saved")


if __name__ == "__main__":
    main()
