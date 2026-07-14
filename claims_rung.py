"""Claims-ladder rung (spec/bullet-route.spec.md step 4): rsearch-dN over a kc+ZCA
linear ckpt, measured by the SAME measure_elo.measure protocol that produced the
1441/1484/1540 claims (SF@1320 anchor, 0.05s/move, alternating colors, ply cap 120,
Wilson-style CI from score SE).

Usage: python claims_rung.py <ckpt.pt> <depth> <games> <agent_name>
Failure modes: ZCA identity gate failure -> SystemExit (corpus_gen.raw_weights);
no stockfish -> measure reports ordinal only (not claims-grade).
"""
import importlib
import sys

import chess

from corpus_gen import raw_weights
from measure_elo import measure


def main():
    ckpt, depth, games, name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    w, b = raw_weights(ckpt)
    srch = importlib.import_module("rsearch4").Searcher(w, b)
    mover = lambda bd: chess.Move.from_uci(srch.search(bd.fen(), depth)[0])
    measure(mover, name, games, merge=20)


if __name__ == "__main__":
    main()
